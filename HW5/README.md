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
Вывод по аугментациям:
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
