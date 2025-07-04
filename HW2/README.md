1.2 Расширение логистической регрессии

Для выполнения задания по многоклассовой классификации и оценке качества модели были внесены следующие изменения:\
Изменения в utils.py:\
  Генерация данных(Функция make_classification_data была модифицирована для генерации многоклассовых данных с использованием sklearn.datasets.make_classification)\
  Функция точности(Функция accuracy была переписана для работы с многоклассовой задачей.)\
  Добавлены метрики(Реализованы метрики: precision, recall, f1_score, roc_auc_score)\
Confusion Matrix:\
Добавлена функция визуализации confusion matrix\
Модель:\
Модель логистической регрессии была адаптирована для многоклассовой классификации.\

![image](https://github.com/user-attachments/assets/2eb23d70-53e9-454c-bd13-d28af1028302)

Модель показывает хорошее качество предсказаний для класса 0, умеренное — для класса 2 и наихудшее — для класса 1.

2.1 Кастомный Dataset класс и 2.2 Эксперименты с различными датасетами\
В рамках этого этапа был реализован собственный класс CustomCSVDataset, предназначенный для автоматизированной загрузки, предобработки и подготовки данных для обучения моделей. Внутри класса были реализованы следующие шаги:\
Загрузка данных из CSV-файла с помощью pandas.read_csv.\
Удаление строк с пропущенными значениями (dropna).\
Разделение на признаки X и целевую переменную y.\
Автоматическое определение категориальных признаков и их кодирование через LabelEncoder.\
Нормализация числовых признаков при помощи StandardScaler.\
Нормализация целевой переменной в случае регрессии.\
Преобразование всех данных в torch.tensor для последующей подачи в модель.\
Далее, в части 2.2, мы провели обучение моделей с использованием подготовленных данных. Были выбраны два разных CSV-датасета:\
Для задачи регрессии: Used Car Price Dataset Extended - https://www.kaggle.com/datasets/therohithanand/used-car-price-prediction?resource=download \
Для задачи бинарной классификации: Water Potability Dataset - https://www.kaggle.com/datasets/adityakadiwal/water-potability \
В случае задачи регрессии модель была обучена на данных о ценах автомобилей. Полученные потери (loss) в ходе 100 эпох демонстрируют стабильное снижение:

![image](https://github.com/user-attachments/assets/1cd1789f-e164-43ed-a343-0ccb9acbd4a5)

Аналогично, в задаче классификации, модель обучалась предсказывать пригодность воды по химическим показателям. Скриншот результатов обучения по классификации также представлен:

![image](https://github.com/user-attachments/assets/b3021f14-9400-40b4-b572-cacfbdf1536a)

Таким образом, был реализован универсальный механизм загрузки и подготовки данных, после чего на его основе проведено обучение моделей как для регрессии, так и для классификации.

3.1
В данном разделе была проведена серия экспериментов по подбору гиперпараметров модели. В частности, варьировались следующие параметры:\
Скорость обучения (learning rate): использовались значения 0.001, 0.01, 0.5\
Размер батча (batch size): тестировались 16 и 32\
Оптимизаторы: применялись SGD, Adam, RMSprop\
На графике ниже представлено сравнение всех комбинаций параметров:\

![image](https://github.com/user-attachments/assets/13fda4fe-2632-4d68-b1d6-8d4d53854223)

Вывод по графику:\
Оптимизаторы Adam и RMSprop при небольших learning rate показывают наилучшую и наиболее стабильную сходимость.\
Слишком большой learning rate = 0.5 приводит к нестабильному поведению и высоким колебаниям loss.\
Разница между batch size 16 и batch size 32 не критична, но меньший батч может давать более плавную динамику, особенно в сочетании с правильным lr и оптимизатором.

3.2 Feature Engineering\
В рамках данного задания была проведена работа по созданию новых признаков с целью улучшения качества модели линейной регрессии.\
В качестве базовой модели использовалась линейная регрессия из раздела 2.2, в которой использовались только исходные признаки.\
Результаты на валидации показали, что модель с расширенными признаками показывает сопоставимое или чуть более высокое значение loss.

![image](https://github.com/user-attachments/assets/a18eada0-709f-40b3-a20f-6f6b351d9bf9)



