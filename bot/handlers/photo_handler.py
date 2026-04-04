"""Handler for photo messages — classifies weather and gives clothing advice.

Flow:
  1. Receive photo from user
  2. Download image bytes from Telegram
  3. Send to ViT classifier → full image analysis
  4. Try Qwen LLM → unique contextual recommendation
  5. If LLM fails, return detailed classifier analysis summary
  6. Return clean text response to user
"""

import logging
import random

from bot.services import llm_client as llm
from bot.services import vit_classifier as classifier

logger = logging.getLogger(__name__)

WEATHER_EMOJI = {
    "sunny": "☀️",
    "cloudy": "☁️",
    "rainy": "🌧️",
    "snowy": "❄️",
    "foggy": "🌫️",
    "night": "🌙",
}

# Detailed, varied fallback recommendations
# Each type has multiple options — chosen randomly based on visual features
_DETAILED_ADVICE = {
    "sunny": [
        "Сегодня жарко и солнечно. Наденьте футболку из хлопка или льна, шорты или лёгкое платье. Обязательно возьмите солнцезащитные очки — яркость высокая. Кепка или панама спасут голову от перегрева. На ноги — сандалии или лёгкие кеды. Не забудьте SPF 30+ на открытые участки кожи.",
        "Ясный солнечный день — одевайтесь максимально легко. Хлопковая футболка и шорты будут в самый раз. Если планируете долго быть на улице, возьмите лёгкую рубашку с длинным рукавом — она защитит руки от загара. Обувь — открытая или дышащая. Вода с собой обязательна.",
        "Солнечно и тепло. Идеальный вариант — футболка и шорты или лёгкое платье. На ноги — сандалии, кроссовки с сеткой или мокасины. Солнцезащитный крем SPF 30+ и очки — мастхэв. Если есть светлая панама или бейсболка, берите — будет прохладнее.",
    ],
    "cloudy": [
        "Небо затянуто облаками, но дождя пока нет. Одевайтесь слоями: футболка + лёгкая кофта или ветровка. На ноги — кроссовки или ботинки. Зонт бросьте в сумку на всякий случай. Если ветер ощутимый, добавьте тонкий шарф на шею.",
        "Пасмурно и прохладно. Многослойность — ваш друг: футболка, флисовая кофта, лёгкая куртка. Так можно снимать или надевать слои по ощущениям. Джинсы или брюки, кроссовки. Зонт в рюкзаке не помешает — погода может измениться.",
        "Облачно, возможен лёгкий дождь. Наденьте водоотталкивающую куртку поверх кофты, джинсы или брюки, удобные кроссовки. Зонтик возьмите — лучше перестраховаться. Если планируете много ходить, добавьте лёгкий шарф.",
    ],
    "rainy": [
        "Идёт дождь — экипируйтесь по максимуму. Водонепроницаемая куртка с капюшоном обязательна. Под неё — кофта или флиска. На ноги — резиновые сапоги или непромокаемые ботинки. Зонт — большой, чтобы укрыл с головой. Сумку замените на рюкзак с водозащитой — обычная промокнет за минуты.",
        "Ливень. Дождевик или плащ с капюшоном — первое дело. Под низ — всё что угодно, но избегайте джинсов (долго сохнут). Лучше спортивные штаны из синтетики. Обувь — резиновая или с мембраной Gore-Tex. Зонт-трость надёжнее складного. Телефон в водонепроницаемый чехол.",
        "Мокро и сыро. Водонепроницаемая куртка, штаны из быстросохнущей ткани, резиновые ботинки. Избегайте светлой одежды — грязь и капли будут видны. Зонт обязательно, а рюкзак лучше затянуть дождевиком. Носки возьмите с запасом — промокнут точно.",
    ],
    "snowy": [
        "Снег и мороз. Одевайтесь как капуста: термобельё на тело, сверху флисовая кофта или свитер, затем тёплый пуховик. Шапка обязательна, шарф закрывает шею и лицо, перчатки или варежки — руки мёрзнут первыми. На ноги — утеплённые ботинки с нескользящей подошвой. Носки — шерстяные.",
        "Зимняя погода, минусовая температура. Верх — пуховик или парка с мехом. Под него — свитер и термобельё. Шапка-ушанка или тёплая вязаная, шарф обмотать вокруг шеи, варежки тёплее перчаток. Брюки утеплённые, обувь — зимние ботинки с протектором от скольжения.",
        "Холодно и снежно. Термобельё, свитер, тёплая куртка с капюшоном — три слоя минимум. Шапка, шарф, варежки — без них никуда. На ноги — валенки, дутики или зимние ботинки с толстой подошвой. Если дорога обледенела, прицепите к обуви ледоступы.",
    ],
    "foggy": [
        "Туман — влажность высокая, проникает до костей. Наденьте водоотталкивающую куртку, флисовую кофту под неё. Шарф закроет шею от сырости. Джинсы или брюки, непромокаемые ботинки. Зонт пригодится — туман часто переходит в мелкий дождь. Видимость низкая, так что светлая одежда сделает вас заметнее для водителей.",
        "Туманно и сыро. Флиска + ветровка — оптимальный набор. Шарф на шею, перчатки тонкие. На ноги — ботинки с водоотталкивающей пропиткой. Туман конденсируется на одежде, так что избегайте впитывающих тканей вроде хлопка. Синтетика сохнет быстрее.",
        "Мгла и сырость. Куртка с мембраной, тёплая кофта, шарф. Перчатки пригодятся — от тумана руки мерзнут. Обувь — закрытая, непромокаемая. Если есть светоотражающие элементы на одежде — отлично, в тумане вас плохо видно. Зонт возьмите лёгкий.",
    ],
    "night": [
        "Ночь — холодно и темно. Тёплая куртка или пуховик, шапка, шарф, перчатки. Под куртку — свитер или флиску. На ноги — утеплённые ботинки. Если светоотражающие элементы есть на одежде — обязательно, в темноте вас плохо видно. Фонарик в телефон не заменит настоящий — лучше возьмите компактный.",
        "Тёмное время суток, температура падает. Пуховик или тёплая парка, шапка закрывает уши, шарф обмотан вокруг шеи. Перчатки или варежки. Термобельё под одежду, если планируете долго быть на улице. Обувь — тёплая, с толстой подошвой (земля холодная). Светлая одежда или жилет со светоотражателями — безопасность.",
        "Ночная прогулка. Куртка потеплее, чем днём — минимум на один слой больше. Шапка, шарф, перчатки обязательны. Под куртку — свитер или худи. Джинсы, тёплые ботинки. Возьмите фонарик или включите фонарик на телефоне. Если есть светоотражатель на куртке — расстегните верх, чтобы его было видно.",
    ],
}


def _clean_response(text: str) -> str:
    """Remove markdown artifacts and ensure clean plain text."""
    text = text.replace("**", "").replace("*", "")
    text = text.replace("__", "").replace("_", "")
    text = text.replace("`", "")
    text = text.replace("<|", "").replace("|>", "")
    import re
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _build_fallback_response(result: dict) -> str:
    """Build a detailed response from classifier analysis when LLM is unavailable.

    Picks one of 3 varied options based on brightness/contrast to add variety.
    """
    weather_type = result["weather_type"]
    confidence = result["confidence"]
    emoji = WEATHER_EMOJI.get(weather_type, "🌤️")

    features = result.get("visual_features", {})
    brightness = features.get("brightness", {})
    color = features.get("color_temperature", {})
    contrast = features.get("contrast", {})
    saturation = features.get("saturation", {})

    # Build detailed analysis string
    details = []
    b_val = brightness.get("brightness_0_255", 0)
    if b_val > 200:
        details.append("очень яркое изображение — вероятно полдень или сильный источник света")
    elif b_val < 45:
        details.append("тёмное изображение — ночное время или плотная облачность")
    elif b_val < 120:
        details.append("приглушённое освещение — раннее утро или вечер")
    else:
        details.append("нормальная освещённость — дневное время")

    warmth = color.get("warmth_ratio", 0)
    blueness = color.get("blueness_ratio", 0)
    if warmth > 0.2:
        details.append("заметны тёплые оранжевые и красные тона — возможно закат или рассвет")
    elif blueness > 0.2:
        details.append("преобладают холодные голубые тона — тень или зимний день")

    c_level = contrast.get("contrast_level", "medium")
    if c_level in ("very_low", "low"):
        details.append("низкий контраст — туман, дымка или пасмурность")
    elif c_level == "high":
        details.append("высокий контраст — чёткая видимость, резкие тени")

    sat = saturation.get("avg_saturation", 0)
    if sat > 0.4:
        details.append("цвета насыщенные и яркие — хорошая видимость")
    elif sat < 0.15:
        details.append("цвета блёклые и серые — тусклое освещение или плотная облачность")

    # Pick varied option based on hash of visual features (deterministic but varied)
    seed = int(b_val * 100 + warmth * 1000 + sat * 500) % 3
    options = _DETAILED_ADVICE.get(weather_type, [
        "Одевайтесь по погоде — куртка, удобная обувь, зонт на всякий случай."
    ])
    advice = options[seed]

    detail_str = ". ".join(details).capitalize()

    return (
        f"{emoji} Погода: {weather_type} (уверенность {confidence:.0%})\n"
        f"Анализ фото: {detail_str}.\n\n"
        f"Рекомендация по одежде:\n{advice}"
    )


async def handle_photo_async(file_bytes: bytes, user_message: str = "") -> str:
    """Process a photo and return a clothing recommendation from LLM.

    Falls back to detailed classifier-based analysis if LLM is unavailable.

    Args:
        file_bytes: Raw image bytes from Telegram.
        user_message: Optional text sent with the photo.

    Returns:
        Clean text recommendation (no markdown, Russian only).
    """
    try:
        # Step 1: Full image analysis from ViT
        result = await classifier.classify_image(file_bytes)
        weather_type = result["weather_type"]
        confidence = result["confidence"]
        context = result["context_for_llm"]

        if user_message:
            context += f"\nUser also said: {user_message}"

        # Step 2: Try LLM for unique advice
        try:
            recommendation = await llm.get_clothing_recommendation(context)
            recommendation = _clean_response(recommendation)
            emoji = WEATHER_EMOJI.get(weather_type, "🌤️")
            return f"{emoji} Погода: {weather_type} (уверенность {confidence:.0%})\n\n{recommendation}"

        except llm.LLMError:
            # Fallback to detailed classifier analysis
            return _build_fallback_response(result)

    except classifier.ClassifierError as e:
        logger.error("Classifier error: %s", e)
        return (
            "Не удалось проанализировать фото. "
            "Убедитесь что это фотография улицы и попробуйте снова."
        )

    except Exception as e:
        logger.error("Unexpected error processing photo: %s", e)
        return (
            "Произошла ошибка при обработке фото. "
            "Попробуйте ещё раз или отправьте другое фото."
        )


def handle_photo() -> str:
    """Sync placeholder for --test mode."""
    return (
        "Фото получено! Анализирую погоду и подбираю одежду...\n\n"
        "Это заглушка — подключите classifier и LLM для работы."
    )
