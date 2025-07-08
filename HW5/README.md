Задание 1: Стандартные аугментации torchvision\
Создайте пайплайн стандартных аугментаций torchvision (например, RandomHorizontalFlip, RandomCrop, ColorJitter, RandomRotation, RandomGrayscale).\
Примените аугментации к 5 изображениям из разных классов (папка train).\
Визуализируйте:\
Оригинал\
Результат применения каждой аугментации отдельно\
Результат применения всех аугментаций вместе

Что я сделал:\
Загрузил датасет из data/train, использован кастомный класс CustomImageDataset.Взял по одному изображению из 5 разных классов.\
Создан список стандартных аугментаций: RandomHorizontalFlip,RandomCrop,ColorJitter,RandomRotation,RandomGrayscale.\
Для каждого изображения: показан оригинал, применена каждая аугментация по отдельности и результат всех аугментаций вместе.\
Вывод по аугментациям:\
RandomHorizontalFlip -отражает изображение по горизонтали. Эффект визуально очевиден.\
![image](https://github.com/user-attachments/assets/eb818f1c-88d8-4d35-9017-0b67a16c190f)

RandomCrop- вырезает случайную часть изображения.\
![image](https://github.com/user-attachments/assets/f7e474bc-d25c-4c07-8da0-0e001c27b485)

ColorJitter -меняет яркость и насыщенность.\
![image](https://github.com/user-attachments/assets/2038fd8b-4acd-43cb-85e0-73dba4144bb9)

RandomRotation -поворачивает изображение, что приводит к появлению чёрных углов.\
![image](https://github.com/user-attachments/assets/68bc6d53-f176-4357-96cd-df701843b2a8)

RandomGrayscale -превращает изображение в чёрно-белое, что может быть полезно для обучения модели инвариантности к цвету.\
![image](https://github.com/user-attachments/assets/d26765e0-2384-46b7-a948-c9d0f3a907c8)

![image](https://github.com/user-attachments/assets/3c12c903-ff6d-4ba1-a67c-9ec1e3a666e8)\
![image](https://github.com/user-attachments/assets/83aa2d9d-47db-48fe-b402-bbed31d7afdc)

В итоговом пайплайне комбинируются все вышеуказанные трансформации.Результат — сильно искажённое изображение.\
![image](https://github.com/user-attachments/assets/d82eef87-7c76-4abd-8187-79a0e206ce7c)

Изображения в папке result/task1.


Задание 2: Кастомные аугментации.\
Реализуйте минимум 3 кастомные аугментации (например, случайное размытие, случайная перспектива, случайная яркость/контрастность).\
Примените их к изображениям из train.\
Сравните визуально с готовыми аугментациями из extra_augs.py.\

Реализовал три пользовательские аугментации изображений.\
RandomBlur:\
Применяет случайное гауссовское размытие к изображению.\
Сначала случайным образом выбирается нечетное значение размера ядра (от 1 до 19).\
Затем к изображению применяется cv2.GaussianBlur с этим ядром.\
RandomSquareBlack\
Затирает случайную квадратную область на изображении.\
Выбирается случайно сторона квадрата, не превышающая наименьшую сторону изображения.Координаты подбираются так, чтобы квадрат полностью помещался.\
И окрашиваем область в черный.\
RandomRectangleBlac.\
Аналогично, выбираем случайно размеры прямоугольника, добавляем проверку на то, чтобы стороны не были равны.\
Для визуального сравнения выбрал одну картинку из тренировочного датасета (Гароу) (Вывел для двух картинок, в отчете скрины для одной картинки).\
На нее поочередно применялись:\
встроенные аугментации из extra_augs.py: AddGaussianNoise, RandomErasingCustom, CutOut;\
и мои кастомные аналоги: RandomBlur, RandomRectangleBlack, RandomSquareBlack.\
Моя кастомная против AddGaussianNoise:\
![image](https://github.com/user-attachments/assets/b1c3ea0c-f6a7-4191-9469-8b476e734502)\
![image](https://github.com/user-attachments/assets/67b4ec32-8615-42d0-96a3-878990573d33)

Моя снижает резкость изображения, делая контуры мягче и размывая детали. Это может имитировать движение камеры или фокусировку.
AddGaussianNoise, наоборот, добавляет высокочастотный шум.

Моя кастомная против RandomErasingCustom:
![image](https://github.com/user-attachments/assets/243681e7-0102-4402-8da3-37b890eb3d41)\
![image](https://github.com/user-attachments/assets/70406449-4b0c-4972-b8ec-caa85b0c2b6f)

Оба затирают часть изображения, но прямоугольник кастомной версии четко контролирует форму и область.

Моя кастомная против CutOut.
![image](https://github.com/user-attachments/assets/c7062c4e-e6ae-4a10-b80c-bb44fb80c197)\
![image](https://github.com/user-attachments/assets/2a77ef92-655e-429d-83e8-9d4c44ae464f)

Эффект аналогичен, но квадрат в кастомке масштабируется случайным образом.
Изображения в папке result/task2.

