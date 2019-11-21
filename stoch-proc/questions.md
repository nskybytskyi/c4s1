1. Означення випадкових процесів. Фазовий простір, параметрична множина, траєкторія випадкового процесу. Порівняння різних означень. Приклади.

    Перше означення: випадковий процес --- сукупність випадкових величин, що задані на одному ймовірносному просторі та індексовані елементами деякої параметричної множини.

    Параметрична множина може бути дискретною (натуральні або цілі числа), або неперервною (дійсні або невід'ємні дійсні. Рідше комплексні числа).

    Фазовий простір --- довільна (вимірна) множина значень випадкового процесу. Також може бути дискретним або неперервним.

    Приклади випадкових процеів: кусково сталий, лінійний (випадкові slope та intercept), сінусоїдальний (випадкові амплітуда і частота).

    Друге означення: функція двох змінних xi(t, omega).

    Такєкторією випадкового процесу називається t mapsto xi(t, omega).

2. Скінченновимірна функція розподілу випадкового процесу. Завдання процесу у широкому розумінні. Стохастична еквівалентність. Теорема Колмогорова.

    Скінченновимірною функцією розподілу називається F_{t_1, t_2, ..., t_n} (A_1, A_2, ..., A_n) = P(xi(t_1) in A_1, xi(t_2) in A_2, ..., xi(t_n) in A_n).

    F_{t_1, t_2, ..., t_n} (x_1, x_2, ..., x_n) = P(xi(t_1) <= x_1, xi(t_2) <= x_2, ..., xi(t_n) <= x_n).

    Випадковий процес називається заданим у широкому розумінні, якщо відомі всі його скінченновимірні функції розподілу.

    Два випадкових процеси стохастично еквівалентні якщо усі їхні скінченновимірні розподілу однакові.

    Теорема Колмогорова: для існування випадкового процесу із заданими скінченновимірними функціями розподілу достатньо, або всі вони (функції розподілу) були неперервні зліва по кожній змінній, аби їх границі  на нескінченності були адекватними, аби вони не залежали від порядку індексів, і аби додавання нескінченних змінних не змінювало їх.

3. Неперервність випадкових процесів за ймовірністю та з ймовірністю 1. Сепарабельність. 

    Випадковий процес неперервний за ймовірністю, якщо forall epsilon > 0: lim_{t to t_0} P(\|xi(t) - xi(t_0)} > epsilon) = 0. Якщо forall t in T xi(t) неперервне то відповідний випадковий процес неперервний за ймовірністю.

    Випадковий процес неперервний у середньому квадратичному якщо lim_{t to t_0} M\|xi(t) - xi(t_0)\|^2 = 0.

    Випадковий процес неперервний з ймовірністю 1 якщо t mapsto xi(t, omega) неперервна при omega in Omega', де P(Omega') = 1.

    З неперервності у середньому квадратичному не випливає неперервність з ймовірністю 1.

    Випадковий процес називється сепарабельним, если для всех a, b, c, d: выполняется условие
        если для всех элементов всюду плотной счётной последовательности {t_n} которые лежат в (c, d) выполняется xi(t_n) в [a, b], то с вероятностью 1 xi(t) в [a, b] для всех t из (c, d)

    Якщо випадковий процес неперервний з ймовірністю 1, то він сепарабельний.

4. Класифікація випадкових процесів.

    За параметричною множиною:
    - Якщо T in R дискретная множина, то випадковий процес з дискретним часом.
    - Якщо T in R неперервна множина, то випадковий процес з неперервним часом.
    - Якщо T in R^n дискретна множина, то дискретне випадкове поле.
    - Якщо T in R^n неперервна множина, то неперервне випадкове поле.

    За фазовим простором:
    - Якщо S скінченна, то випадковий вектор.
    - Якщо S дискретна, то випадковий процес з дискретними значеннями.
    - Якщо S неперервна, то випадковий процес з неперервними значеннями.

    За ймовірносною характеристикою:
    - З незалежними у сукупності значеннями.
    - З незалежними приростами.
    - Марківські (не мають пам'яті).

5. Характеристики випадкових процесів: математичне сподівання, дисперсія, кореляційна функція, нормована кореляційна функція.

    Математичне сподівання --- функція m_xi t mapsto M xi(t). Ця функція зовсім не випадкова. Математичне сподівання лінійне. З-під нього можна виносити невипадкову функцію.

    Дисперсія --- функція D_xi mapsto D xi(t) = M (xi(t) - m_xi(t))^2. Також не випадкова функція. Невід'ємна. З-під неї можна виносити константи з квадратом. Дисперсія суми незалежних -- сума дисперсій.

    Кореляційна функція --- K: (t_1, t_2) mapsto M [(xi(t_1) - m_xi(t_1)) * (xi(t_2) - m_xi(t_2))]. Кореляційна функція симетрична. З-під неї можна виносити невипадкові функції. Обмежена за модулем коренем з добутку дисперсій.

    Нормована кореляційна функція --- кореляційна функція розділена на корінь з добутку дисперсій.

6. Взаємна кореляційна функція. Кореляційна функція суми випадкових процесів.

    Взаємною кореляційною функцією двох випадкових процесів називається K_{xi, eta}(t_1, t_2) = M [normed xi(t_1) normed eta(t_2)].

    Кореляційна функція процесу eta(t) = sum(xi_i(t) for i in range(n)) може бути обчислена за формулою K_eta(t_1, t_2) = sum(K_{xi_i}(t_1, t_2) for i in range(n)) + sum(K_{xi_i xi_j}(t_1, t_2) for i != j in range(n)).

7. Комплексно-значні випадкові процеси та їх характеристики.

    Комплекснозначний випадковий процес: theta(t) = xi(t) + i eta(t).

    Матсподівання m_theta(t) = m_xi(t) + i m_eta(t). Інші властивості такі ж як у звичайного матсподівання.

    Дисперсія D theta(t) = D xi(t) + i D eta(t). Інші властивості такі ж як у звичайної дисперсії, окрім того, що невипадкові комплексні множники виносяться з-під неї не як квадрат а зі спряженням.

8. Властивості збіжності у середньому квадратичному. Неперервність та похідна випадкового процесу у середньому квадратичному. Математичне сподівання та кореляційна функція похідної.
    
    Послідовність випадкових величин xi_n збігається до xi у середньому квадратичному якщо lim_{n to infty} M[(xi_n(t) - xi(t))^2] = 0.

    Випадковий процес неперервний якщо forall tau lim_{t to tau} M[\|xi(t) - xi(tau)\|^2] = 0.

    Похідна випадкового процесу --- такий випадковий процес dot xi(tau), що lim_{t to tau} M[\|(xi(t) - xi(tau))/(t - tau) - dot xi(tau)\|] = 0.

    Матсподівання похідної випадкового процесу --- похідна матсподівання випадкового процесу.

    Кореляційна функція похідної випадкового процесу --- мішана похідна кореляційної функції випадкового процесу.

9. Інтеграл випадкового процесу у середньому квадратичному. Математичне сподівання та кореляційна функція інтегралу випадкового процесу.

    1/T int(xi(t) dt for t in [0, T])

    Матсподівання інтегралу випадкового процесу --- інтеграл матсподівання випадкового процесу.

    Кореляційна функція інтегралу випадкового процесу --- подвійний інтеграл кореляційної функції випадкового процесу.

10. Вінерівський випадковий процес. Властивості вінерівського процесу. 

    W вінерівський якщо:
    - P(W(0) = 0) = 1;
    - прирости незалежні;
    - процес однорідний;
    - траєкторії неперервні з ймовірністю 1.

    Властиовсті:
    - Його одновимірний розподіл --- нормальний.
    - Його матсподівання 0, дисперсія лінійна.
    - Є ще процес з дрифтом (його матсподівання лінійне).
    - Його коваріаційна функція --- min(t_1, t_2).

    Повторний логаритм: lim sup_{t to infty} W_t / sqrt(2 t ln ln t) = 1 almost surely.

11. Гауссівські випадкові процеси. Двовимірні гауссівські процеси. 

    Всі скінченновимірні щільності мають нормальний розподіл відповідної розмірності.

    ... [щось про двовимірні гаусівські процеси]

12. Основні поняття статистичного моделювання. Методи та алгоритми моделювання гауссівських випадкових процесів та вінерівського процесу.

    Генеруємо випадкові числа, рівномірно розподілені на [0, 1].

    Застосовуємо обернену функцію розподілу.

    ... [щось ще]

13. Однорідний випадковий процес Пуассона. Розподіл, властивості, різні означення. Узагальнення випадковий процес Пуассона.

    Перше означення:
    - P(xi < x) = 1 - e^{-lambda x}, for x >= 0;
    - f(x) = lambda e^{-lambda x}, for x >= 0;
    - M xi^k = k! / lambda^k;
    - відсутність післядії: P(xi > t + x | xi > x) = e^{-lambda t};

    Друге означення:
    - P(xi(0) = 0) = 1;
    - прирости незалежні;
    - xi(t) ~ e^{-lambda t};

    Траєкторії кусково сталі.

    Узагальнений випадковий процес Пуассона --- стрибки різні.

    Властивості:
    - Неперервний у середньому квадратичному і за ймовірністю.
    - Не має похідної у середньому квадратичному. 
    - Має інтеграл у середньому квадратичному.

14. Стаціонарні випадкові процеси в широкому та вузькому розумінні. Характеристики. Властивості кореляційної функції. Приклади.

    Випадковий процес стаціонарний у вузькому розумінні, якщо всі скінченновимірні функції розподілу не змінюються при зсуві на tau.

    Випадковий процес стаціонарний у широкому розумінні якщо його матсподівання не залежить від часу, а кореляційна функція залежить лише від різниці своїх аргументів.

    Зі стаціонарності у вузькому розумінні випливає стаціонарність у широкому розумінні, але не навпаки. Для гауссівських процесів ці дві стаціонарності еквівалентні.

    Приклади: zeta(t) = xi cos(t) + eta sin(t), xi, eta ~ N(0, 1) (у широкому але не у вузькому).

    Властивості кореляційної функції такі ж як і завжди окрім того що тепер вона від одного аргументу.

15. Спектральна теорія стаціонарних процесів. Скінченний дискретний спектр.

    Можна записати xi(t) = sum(A_i sin(omega_i t + phi_i) for i in range(n)).

    Тоді скінченний спектр --- сукупність пар (omega_i, D_i), i in range(n):
    -omega_i --- частота гармоніки;
    -D_i --- дисперсія гармоніки;

16. Спектральна теорія стаціонарних процесів. Зліченний дискретний спектр. Неперервний спектр стаціонарного процесу.
    
    Можна записати xi(t) = sum(A_i sin(omega_i t + phi_i) for i in N).

    Тоді зліченний дискретний спектр --- послідовність пар (omega_i, D_i), i in N:
    -omega_i --- частота гармоніки;
    -D_i --- дисперсія гармоніки;

    Спектральна щільність --- обернене перетворення Фур'є від автокореляційної функції.

17. Ергодичність випадкових процесів та стаціонарних процесів.

    Випадковий процес зі сталим матсподіванням називається ергодичним, якщо інтеграл у середньому квадратичному збігається до матсподівання за ймовірністю про T to infty.

    Випадковий процес зі сталим матсподіванням є ергодичним тоді і тільки тоді, коли K(t_1, t_2) to 0 при \|t_1 - t_2\| to infty.

    Теорема Слуцького: якщо дійсна частина стаціонарного у широкому розумінні комплекснозначного випадкового процесу є ергодичною, то і він увесь ергодичний.

    Це була ергодичність відносно матсподівання, а є ще ергодичність відносно кореляційної функції: 1/T int(xi(t + tau) conj eta(t) d t for t in [0, T]) to K(tau) при T to infty.

18. Узагальнення інтегралу у середньому квадратичному для L_2-процесів.

    L2-процес --- скінченне матсподівання квадрату.

    Всі випадкові процеси наближаємо кусково-сталими L_2 процесами, щоб можна було проінтегрувати як у курсі класичного аналізу.

    Інтеграл --- границя послідовності квадратурних сум.

    Збіжність потрібна у середньому квадратичному.

19. Стохастичний інтеграл за вінерівським процесом.

    Стохастичний інтеграл --- ніби як інтеграл Рімана-Стільт'єса.

    Всі випадкові процеси наближаємо кусково-сталими L_2  процесами, щоб можна було проінтегрувати як у курсі класичного аналізу. Збіжність потрібна у середньому квадратичному.

    Стохастичний інтеграл за вінерівським процесом:
    - I_0 = lim_{n to infty} sum(omega(t_i) (omega(t_{i + 1}) - omega(t_i)) for i in range(n)) (Іто)
    - I_1 = lim_{n to infty} sum(omega(t_{i + 1}) (omega(t_{i + 1}) - omega(t_i)) for i in range(n)).
    - M[I_1 - I_0] = T

20. Стохастичний диференціал. Формула Іто. Приклади застосувань. 

    ... [щось про стохастичний диференціал]

    Формула Іто --- формула заміни змінних у стохастичному диференціальному рівнянні.

    Застосовується, як не складно здогадатися, у стохастичних диференціальних рівняннях.