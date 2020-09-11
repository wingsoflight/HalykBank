# 1. Развертывание среды python
1. Нужно установаить модуль pinenv "pythom -m pip install pipenv"
2. Для того чтобы создать среду в текущей папке нужно определить системную переменную "set PIPENV_VENV_IN_PROJECT="enabled" для Windows
3. Инициируем виртуальную среду "pipenv shell"
4. Устанавливаем нужные модули "pipenv install pymupdf"
# 2. Чтение данных из PDF файла Python
В файле readpdf.py описан скрипт который считывает данные из test.pdf
# 3. Мини-приложение на C# (simple app input output)
В файле Program.cs исходный код консольного приложения который считывают строку и выводит ее на экран.
Скомпилированный вариант в папке ./bin/Release
# 4. Парсинг текста на SQL
В файле string_parse_split.sql пример разделения строки на сегменты разделенные точками.
# 5. Распознавание наличия живой подписи на скан документах
1. Установите нужные модули через команду "pipenv install"
2. Запустите скрипт detect_handsign.py и передайте путь к pdf файлу в аргументах "py detect_handsign.py example.pdf"
3. Если скрипт обнаружит подпись то откроется окно opencv, нажмите любую клавишу чтобы закрыть его.
Метод основан на данной статье https://www.researchgate.net/publication/326271482_Detecting_handwritten_signatures_in_scanned_documents
