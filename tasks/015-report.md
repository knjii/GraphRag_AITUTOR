# Отчёт по задаче 015: проверка источников библиотеки

Проверено 2026-09-17. Изменён только `tasks/015-report.md`; манифест и чужие изменения не затронуты, коммитов нет. Результаты записывались в этот файл по ходу работы, затем сведены в таблицы ниже.

Сеть использована только для чтения через Python `urllib.request`. HEAD выполнялся с переходами по редиректам. Тела GET и распакованные PDF-потоки обрабатывались только в памяти: книги, HTML, JSON, TeX и исполняемые файлы на диск не сохранялись. Пакеты не устанавливались.

## Все 9 книг: адреса, коды и размеры

Размеры в байтах. Для восьми одиночных PDF указан полный `Content-Length` ответа HEAD. Для Соколова адрес ведёт на каталог HTML без `Content-Length`; размер комплекта определён по 12 файлам API и независимо сверенным полным размерам в `Content-Range`.

| ID и проверенный адрес | HEAD, конечный код | Полный размер | Content-Type |
|---|---:|---:|---|
| [en-ml-murphy](https://github.com/probml/pml-book/releases/latest/download/book1.pdf) | 200 | 92 323 940 | `application/octet-stream` |
| [en-rl-sutton-barto](http://incompleteideas.net/book/RLbook2020.pdf) | 200 | 73 129 769 | `application/pdf` |
| [en-la-axler](https://linear.axler.net/LADR4e.pdf) | 200 | 2 795 156 | `application/pdf` |
| [en-prob-grinstead-snell](https://math.dartmouth.edu/~prob/prob/prob.pdf) | 200 | 2 993 135 | `application/pdf` |
| [en-anchor-mml](https://mml-book.github.io/book/mml-book.pdf) | 200 | 17 565 898 | `application/pdf` |
| [ru-ml-sokolov](https://github.com/esokolov/ml-course-hse/tree/master/2019-fall/lecture-notes) | 200 | 1 920 743 (сумма 12 PDF; не размер HTML) | `text/html; charset=utf-8` |
| [ru-rl-ivanov](https://arxiv.org/pdf/2201.09746) | 200 | 24 006 224 | `application/pdf` |
| [ru-la-gelfand](https://mccme.ru/free-books/linalg/gelfand.pdf) | 200 | 1 343 897 | `application/pdf` |
| [ru-prob-chernova](http://tvims.nsu.ru/chernova/tv/tv_nsu07.pdf) | 200 | 1 265 657 | `application/pdf` |

Сломанных адресов среди шести `ready` не обнаружено. Это проверка доступности и заголовков, а не полная проверка содержания, лицензий и редакций этих шести книг.

- Murphy: переход с GitHub на `release-assets.githubusercontent.com`, конечный ответ 200. Временный подписанный адрес не предлагается записывать в манифест. `application/octet-stream` — тип вложения GitHub; HEAD не подтверждает PDF-сигнатуру. `Last-Modified: Fri, 18 Apr 2025 10:49:46 GMT`.
- Гельфанд: исходный адрес перенаправляет на `https://old.mccme.ru//free-books//linalg/gelfand.pdf`; конечный HEAD 200. Это рабочий редирект.
- Остальные семь HEAD вернули 200 на исходных адресах без изменения конечного URL.

## en-anchor-mml — адрес и версия подтверждены

Источник PDF: https://mml-book.github.io/book/mml-book.pdf

HEAD 200, 17 565 898 байт, `application/pdf`. GET с `Range: bytes=0-204799` вернул 206, `Content-Range: bytes 0-204799/17565898`, тело 204 800 байт; начало `%PDF-1.5`.

[Страница книги](https://mml-book.github.io/) отвечает GET 200. На ней указано издание Cambridge University Press, апрель 2020. Основная ссылка ведёт на обновляемый вариант PDF с исправлениями; отдельно дан вариант, соответствующий печатному изданию. Даты текущей правки основного PDF на странице не найдено. В колонтитулах самого PDF указана дата редакции **2024-01-15**; HTTP `Last-Modified: Thu, 13 Mar 2025 09:06:18 GMT` — другая дата, её нельзя подменять датой редакции.

Для условий и даты потребовался дополнительный диапазон `bytes=7854859-8117002` (262 144 байта, HTTP 206), после большой обложки: начало текстовых страниц найдено по `/E 7854859` в словаре линеаризации. Flate-потоки распакованы в памяти. Также выполнялись диагностические чтения префикса и диапазона до байта 2 097 151; полного PDF не читали.

Условия использования из PDF, нижний колонтитул страницы i (одна короткая цитата; пробелы восстановлены между фрагментами PDF-текста):

> This version is free to view and download for personal use only. Not for re-distribution, re-sale, or use in derivative works.

Вывод: адрес ведёт на PDF нужной книги, версия установлена. Условия строже простой пометки о бесплатном доступе: приведённая цитата ограничивает просмотр и скачивание личным использованием и отдельно перечисляет запреты. Разрешение на создание производных материалов из этой проверки не следует.

Предложение: `status: ready` в смысле подтверждённого прямого PDF; URL оставить. В `notes` записать дату редакции 2024-01-15 и ограничения; формулировку `license` дополнить запретами распространения, перепродажи и производных работ. Статус доступности не является разрешением на использование книги для конкретного RAG/RL-сценария. Соответствие английской редакции русскому MML из индекса не проверялось.

## ru-ml-sokolov — PDF есть ко всем лекциям каталога

[Каталог из манифеста](https://github.com/esokolov/ml-course-hse/tree/master/2019-fall/lecture-notes): HEAD 200, `text/html; charset=utf-8`, `Content-Length` отсутствует. Единого PDF книги по этому адресу нет.

[GitHub Contents API](https://api.github.com/repos/esokolov/ml-course-hse/contents/2019-fall/lecture-notes?ref=master): GET 200, `Content-Length: 32219`; прочитано 32 219 байт JSON. Это размер описания каталога, не книги. В каталоге 32 файла: 12 PDF, 12 одноимённых TeX, 7 EPS и один STY. Каждой из лекций 01–12 соответствует готовый PDF. Искать PDF той же редакции в других местах репозитория не потребовалось.

Полный список файлов и размеры из API; ссылки ведут на конкретные исходные файлы:

| Файл | Размер, байт |
|---|---:|
| [descent.eps](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/descent.eps) | 605 709 |
| [lecture01-intro.pdf](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture01-intro.pdf) | 127 732 |
| [lecture01-intro.tex](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture01-intro.tex) | 36 642 |
| [lecture02-linregr.pdf](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture02-linregr.pdf) | 169 668 |
| [lecture02-linregr.tex](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture02-linregr.tex) | 45 084 |
| [lecture03-linregr.pdf](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture03-linregr.pdf) | 229 706 |
| [lecture03-linregr.tex](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture03-linregr.tex) | 30 700 |
| [lecture04-linclass.pdf](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture04-linclass.pdf) | 212 352 |
| [lecture04-linclass.tex](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture04-linclass.tex) | 46 273 |
| [lecture05-linclass.pdf](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture05-linclass.pdf) | 111 656 |
| [lecture05-linclass.tex](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture05-linclass.tex) | 23 507 |
| [lecture06-linclass.pdf](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture06-linclass.pdf) | 175 576 |
| [lecture06-linclass.tex](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture06-linclass.tex) | 40 356 |
| [lecture07-trees.pdf](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture07-trees.pdf) | 158 881 |
| [lecture07-trees.tex](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture07-trees.tex) | 40 626 |
| [lecture08-ensembles.pdf](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture08-ensembles.pdf) | 234 289 |
| [lecture08-ensembles.tex](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture08-ensembles.tex) | 41 998 |
| [lecture09-ensembles.pdf](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture09-ensembles.pdf) | 151 838 |
| [lecture09-ensembles.tex](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture09-ensembles.tex) | 51 480 |
| [lecture10-ensembles.pdf](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture10-ensembles.pdf) | 113 732 |
| [lecture10-ensembles.tex](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture10-ensembles.tex) | 20 807 |
| [lecture11-unsupervised.pdf](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture11-unsupervised.pdf) | 108 756 |
| [lecture11-unsupervised.tex](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture11-unsupervised.tex) | 22 147 |
| [lecture12-factorizations.pdf](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture12-factorizations.pdf) | 126 557 |
| [lecture12-factorizations.tex](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/lecture12-factorizations.tex) | 28 871 |
| [plot_roc_001.eps](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/plot_roc_001.eps) | 2 922 672 |
| [precision_recall_harmonic.eps](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/precision_recall_harmonic.eps) | 39 983 |
| [precision_recall_min.eps](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/precision_recall_min.eps) | 36 310 |
| [reg.eps](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/reg.eps) | 19 356 |
| [threshold-approx.eps](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/threshold-approx.eps) | 30 410 |
| [underfitting_overfitting.eps](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/underfitting_overfitting.eps) | 4 259 612 |
| [vkCourseML.sty](https://raw.githubusercontent.com/esokolov/ml-course-hse/master/2019-fall/lecture-notes/vkCourseML.sty) | 18 929 |

**Сумма 12 PDF: 1 920 743 байт.**

Каждый из 12 прямых PDF дополнительно проверен GET с `Range: bytes=0-15`: все ответы **206**, `Content-Length: 16`, начало **`%PDF-1.4`**; знаменатель `Content-Range` совпал с размером соответствующего файла в таблице. Ветки `master` могут меняться; набор относится к каталогу `2019-fall` на дату проверки. Побайтовое соответствие PDF и TeX не проверялось.

Темы проверены по заголовкам разделов всех 12 TeX-файлов, ссылки приведены в таблице:

1. Постановки задач машинного обучения и примеры.
2. Линейная регрессия, функции ошибки, градиентный и стохастический спуск.
3. Переобучение, оценка качества, регуляризация, гиперпараметры, разреженные модели, квантильная регрессия и преобразования признаков.
4. Линейная классификация и метрики, матрица ошибок, AUC.
5. Логистическая регрессия и метод опорных векторов.
6. Многоклассовая и многометочная классификация, категориальные и текстовые признаки.
7. Решающие деревья, критерии разбиений, остановка, обрезка, обработка пропусков.
8. Бутстрап, разложение ошибки на смещение и дисперсию, бэггинг, случайный лес, OOB.
9. Градиентный бустинг, функции потерь, регуляризация, деревья, методы второго порядка.
10. XGBoost и стекинг.
11. Кластеризация, K-Means, графовые и иерархические методы, визуализация, представления.
12. Снижение размерности и PCA, рекомендательные системы, коллаборативная фильтрация, скрытые факторы, неявные оценки и контентные модели.

Предложение: удалить из `notes` предположение об отсутствии PDF и необходимости сборки TeX. Указать комплект из 12 PDF по прямым адресам таблицы. **Оставить `status: check` для нынешней единственной записи с URL каталога**, поскольку в манифесте `ready` определён как известный прямой PDF. После представления комплекта отдельными PDF-записями либо явным списком файлов можно поставить `ready` для этих файлов. Заменять URL комплекта на одну лекцию нельзя: это сократит материал. Поддержка скачивания набора файлов не входила в задачу; лицензию репозитория заново не проверяли.

## ru-la-gelfand — текстовый слой вероятен

Источник: https://mccme.ru/free-books/linalg/gelfand.pdf

HEAD после редиректа: 200, 1 343 897 байт, `application/pdf`. GET с `Range: bytes=0-204799` вернул 206, `Content-Range: bytes 0-204799/1343897`, `Content-Length: 204800`. Прочитано ровно 204 800 байт в память, начало `%PDF-1.4`.

Анализ префикса:

- В сырых байтах нет `/Font`, `/ToUnicode` и `/Subtype /Image`. Случайные совпадения Tj/TJ внутри сжатых данных не использованы как доказательство.
- Удалось распаковать 75 полных Flate-потоков. В них найдены 173 оператора `BT`, 20 917 `Tj`, 5 271 `TJ` и 12 120 команд выбора шрифта `Tf`.
- Примеры ссылок на шрифты в командах: `/R8 0.12 Tf`, `/R20 0.12 Tf`, `/R45 0.12 Tf`, `/R68 0.12 Tf`, `/R79 0.12 Tf`. Это свидетельство использования шрифтов в текстовых потоках; сами словари ресурсов и таблицы соответствия символов в исследованном префиксе не обнаружены.

Вывод: **текстовый слой вероятен**. Префикс содержит текстовые команды со ссылками на шрифты, а не только изображения. Это не доказывает корректное извлечение Unicode, качество формул или наличие текста на каждой странице; счётчики получены диагностическим поиском по распакованным потокам, без полного PDF-парсера. Полный файл и страницы визуально не исследовались, номер издания независимо не подтверждён.

Предложение: `status: ready`, URL оставить (редирект работает). В `notes` заменить предположение о скане результатом проверки: текстовый слой вероятен, первые 200 КиБ содержат BT/Tj/TJ/Tf; качество извлечения кириллицы и формул отдельно проверить на 10 страницах перед индексацией. Условия лицензии этой книги в задаче не перепроверялись.

## Предложения для шести ready

Для каждой строки ниже проверенный адрес, HTTP-код, размер и тип приведены в первой таблице. Изменения только предлагаются, манифест не редактировался.

| ID | Статус | Адрес | Предлагаемая заметка |
|---|---|---|---|
| en-ml-murphy | Оставить ready | Оставить | HEAD 200, 92 323 940 байт; вложение GitHub application/octet-stream. Ссылка latest подвижна, редакцию перед загрузкой сверить; Last-Modified совпадает с датой 2025-04-18 в названии, но это не проверка титула. |
| en-rl-sutton-barto | Оставить ready | Оставить | HEAD 200, 73 129 769 байт, application/pdf. |
| en-la-axler | Оставить ready | Оставить | HEAD 200, 2 795 156 байт, application/pdf. |
| en-prob-grinstead-snell | Оставить ready | Оставить | HEAD 200, 2 993 135 байт, application/pdf. |
| ru-rl-ivanov | Оставить ready | Оставить | HEAD 200, 24 006 224 байт, application/pdf. URL без номера версии; конкретная редакция этим HEAD не подтверждена. |
| ru-prob-chernova | Оставить ready | Оставить | HEAD 200, 1 265 657 байт, application/pdf. |

## Проверки и ограничения

Выполнены сетевые проверки, описанные выше, и локальные проверки согласованности результатов. Исходный код не менялся. `pytest` не запускался: задача содержит только проверку источников и разрешает менять единственный Markdown-отчёт; тестовых файлов для этой задачи нет. Вывод локальных проверок будет дописан ниже.

Не сделаны: скачивание полных книг, визуальная проверка страниц, полная проверка PDF-парсером, проверка качества OCR/формул, юридическая оценка использования материалов. Все использованные внешние источники перечислены прямыми ссылками выше; сторонние поисковые источники не использовались.

Локальная проверка выполнена через `C:\python\rag_textbook\.venv\Scripts\python.exe -` (проверки assert, без сетевых повторов). Вывод дословно:

```text
PASS: all 9 manifest books have source URL, HTTP 200 and size in the summary table
PASS: complete catalog has 32 files and 12 matching PDF/TeX pairs
PASS: sum of 12 lecture PDF sizes is 1920743 bytes
PASS: all 3 check books have evidence and proposed manifest changes
PASS: one short quotation, valid text, no temporary signed URL
5 checks passed
```
