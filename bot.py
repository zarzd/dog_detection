import logging
from aiogram import Bot, Dispatcher, types
from ultralytics import YOLO
import cv2
import os
from config import settings
from aiogram import F
import asyncio
from aiogram.filters.command import Command
from aiogram.types import FSInputFile

logging.basicConfig(level=logging.INFO)
bot = Bot(token=settings.telegram_token)
dp = Dispatcher()


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.reply("Привет! Отправьте мне фотографию или видео с собакой для аннотации.")


@dp.message(F.text)
async def text_message(message: types.Message):
    await message.reply("Отправь мне фотографию или видео, чтобы я сделал аннотацию")


@dp.message(F.photo)
async def handle_photo(message: types.Message):
    try:
        model = YOLO('runs/detect/train/weights/best.pt')
        file_path = f"telegram/{message.photo[-1].file_id}.jpg"
        await message.reply("Скачиваю фотографию...")
        await bot.download(message.photo[-1], destination=file_path)

        await message.reply("Обрабатываю фотографию...")
        results = model(file_path, show=False)
        for result in results:
            result.save(filename='telegram/result.jpg')

        image_from_pc = FSInputFile(f"telegram/result.jpg")
        await message.reply_photo(image_from_pc)
        os.remove(file_path)
        os.remove("telegram/result.jpg")
    except Exception as e:
        logging.error(f"Произошла ошибка: {e}")
        await message.reply("Произошла ошибка при обработке видео.")


@dp.message(F.animation)
async def handle_video(message: types.Message):
    try:
        model = YOLO('runs/detect/train/weights/best.pt')
        # Получаем информацию о видео
        animation = message.animation
        file_id = animation.file_id
        file_path = f"telegram/{file_id}.mp4"

        # Скачиваем видео
        await message.reply("Скачиваю видео...")
        await bot.download(animation, file_path)

        # Обработка видео с помощью OpenCV
        cap = cv2.VideoCapture(file_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        annotated_video_path = f'telegram/annotated_{file_id}.mp4'
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        await message.reply("Обрабатываю видео...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Здесь вы должны заменить model.track() на ваш код обработки видео
            results = model.track(frame, persist=True, show=False)
            annotated_frame = results[0].plot()

            out.write(annotated_frame)

        cap.release()
        out.release()

        # Отправляем аннотированное видео
        await message.reply_animation(FSInputFile(annotated_video_path))

        # Удаляем временные файлы
        os.remove(file_path)
        os.remove(annotated_video_path)

    except Exception as e:
        logging.error(f"Произошла ошибка: {e}")
        await message.reply("Произошла ошибка при обработке видео.")


@dp.message(F.video)
async def handle_video(message: types.Message):
    try:
        model = YOLO('runs/detect/train/weights/best.pt')
        # Получаем информацию о видео
        animation = message.video
        file_id = animation.file_id
        file_path = f"telegram/{file_id}.mp4"

        # Скачиваем видео
        await message.reply("Скачиваю видео...")
        await bot.download(animation, file_path)

        # Обработка видео с помощью OpenCV
        cap = cv2.VideoCapture(file_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        annotated_video_path = f'telegram/annotated_{file_id}.mp4'
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        await message.reply("Обрабатываю видео...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Здесь вы должны заменить model.track() на ваш код обработки видео
            results = model.track(frame, persist=True, show=False)
            annotated_frame = results[0].plot()

            out.write(annotated_frame)

        cap.release()
        out.release()

        # Отправляем аннотированное видео
        await message.reply_video(FSInputFile(annotated_video_path))

        # Удаляем временные файлы
        os.remove(file_path)
        os.remove(annotated_video_path)

    except Exception as e:
        logging.error(f"Произошла ошибка: {e}")
        await message.reply("Произошла ошибка при обработке видео.")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
