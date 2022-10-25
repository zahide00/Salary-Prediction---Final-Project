# EDA
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Hypothesis
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import chi2_contingency, chi2
import time


# Data Preprocesing
from sklearn.model_selection import train_test_split

# Modeling Regression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


# Plots settings
sns.set(rc={'figure.figsize': [15, 7]}, font_scale=1.2) # Standard figure size for all
sns.set_style("whitegrid")


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv(r"C:\Users\arsla\PycharmProjects\pythonProject\datasets\ds_salaries.csv")


# ------------------------------ 1.) KEŞİFCİ VERİ ANALİZİ-----------------------------

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("##################### Describe #####################")
    print(dataframe.describe().T)

check_df(df)


# 1.a) Numerik ve kategorik değişkenleri yakalayınız.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    print(f'cat_cols: {(cat_cols)}')
    print(f'num_cols: {(num_cols)}')
    print(f'cat_but_car: {(cat_but_car)}')
    print(f'num_but_cat: {num_but_cat}')

    return cat_cols, num_cols, cat_but_car, num_but_cat

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)



# Kategorik değişken analiz;
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()



for col in cat_cols:
    cat_summary(df,col,plot=True)


# Nümerik değişken analiz;
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)


# 1.b) Aykırı gözlem analizi yapınız.
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
        quartile1 = dataframe[col_name].quantile(q1)
        quartile3 = dataframe[col_name].quantile(q3)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    print(col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


#Hedef değişken aykırı gözlem analiz
check_outlier(df, "salary_in_usd")


# 1.c) Eksik değer analizi
df.isnull().sum()  #Eksik değer yoktur.

df.head()
# --------------------------------------2.)Veri Önişleme--------------------------------------

# Dolar bazında maaşların olduğu salar_usd değişkeni yeterli olduğu için diğer değişkenler silinmiştir.
df.drop(['Unnamed: 0', 'salary', 'salary_currency'], axis=1, inplace=True)

# remote değişkeni oranlara görenlere göre isimlendirilmiştir.
df["remote_ratio"].replace([0, 50, 100], ["On-site", "Hybrid", "Remote"], inplace=True)

# work_year veri türünü int'den dizeye değiştirilir.
df['work_year'] = df['work_year'].astype(str)
df['work_year'].dtypes
# Remote_ratio veri türünü int'den dizeye değiştirin

df['remote_ratio'] = df['remote_ratio'].astype(str)
df['remote_ratio'].dtypes



grab_col_names(df)






# -----------------------------------3.)Features Engineering---------------------------------------
# Şirket lokasyonlarına bağlı olarak yeni sütün ekleme(Gelişimiş/Gelişmekte olan)

# lst_continent: Company_location'ın ait olduğu kıtanın adını bulunduran değişken
# lst_Development : Şirketin bulunduğu ülkenin gelişmişlik seviyesini bulunduran değişken
lst_europe = ['DE', 'GB', 'HU', 'FR', 'GR', 'NL', 'AT', 'ES', 'PT', 'DK', 'IT', 'HR', 'LU', 'PL', 'RO',
             'BE', 'UA', 'RU', 'MT', 'MD', 'SI', 'CH', 'CZ', 'EE', 'IE']
lst_asia = ['JP', 'IN', 'PK', 'CN', 'AE', 'SG', 'IQ', 'IL', 'IR', 'VN', 'TR', 'MY']
lst_north_america = ['US', 'MX', 'CA', 'AS']
lst_south_america = ['BR', 'CL', 'CO']
lst_africa = ['HN', 'NG', 'KE', 'DZ']
lst_oceanic = ['NZ', 'AU']
#--------------------------------------------------------------------------------------------------------
lst_developed = ['DE', 'JP', 'GB', 'US', 'HU', 'NZ', 'FR', 'GR', 'NL', 'CA', 'AT', 'ES', 'PT', 'DK', 'IT',
                'LU', 'PL', 'BE', 'IL', 'RU', 'MT', 'SI', 'CH', 'TR', 'CZ', 'EE', 'AU', 'IE']
lst_continent = []
lst_developed_countries = []

for s in df.company_location:
    count = 0
    if s in lst_europe:
        lst_continent.append('Europe')
    if s in lst_asia:
        lst_continent.append('Asia')
    if s in lst_north_america:
        lst_continent.append('North America')
    if s in lst_south_america:
        lst_continent.append('South America')
    if s in lst_africa:
        lst_continent.append('Africa')
    if s in lst_oceanic:
        lst_continent.append('Oceanic')
#------------------------------------------------------------
    if s in lst_developed:
        lst_developed_countries.append('developed')
    if s not in lst_developed:
        lst_developed_countries.append('developing')
#------------------------------------------------------------
df['Continent'] = lst_continent
df['Developed/Developing'] = lst_developed_countries
df.head()




# job_title değişkeni baz alınarak "main_role" değişkeni oluşturulur.


# lst_role: rolün ana adını saklar
# lstrole:  job_title da bulunan ünvanların isimlere göre gruplanması

lst_role = ['Scientist', 'Analyst', 'Engineer', 'Manager', 'Lead', 'Director', 'Architect',
            'Principal', 'Developer', 'Consultant', 'Specialist', 'Head' , 'Researcher']
lst_role_ = []
for s in df.job_title:
    s = s.split(' ')
    count = 0
    for ch in s:
        if ch in lst_role:
            lst_role_.append(ch)
            break
df['main_role'] = lst_role_
df.head(30)


# job_title değişkeni baz alınarak "main_field" değişkeni oluşturulur.
# lst_ekonomik: ekonomik alana ait olan rolün adını saklayın
# lstekonomik: iş_unvanı ekonomik alana aitse, ekonomiktir, değilse: AI ve Veri

lst_economic = ['Marketing', 'BI', 'Finance', 'Financial', 'Business', 'Product']
lst_economic_ = []

for s in df.job_title:
    count = 0
    for ch in lst_economic:
        if ch in s:
            count = 1
            break;
    if (count == 1):
        lst_economic_.append('Economic')
    else :
        lst_economic_.append('AI and Data')
df['main_field'] = lst_economic_
df.head()


# Job title değişkeni dönüştürüldükten sonra veriden silinebilir.
df.drop("job_title",axis=1, inplace=True)




# Salary değişkenin ortalamaya göre gruplanması ve yeni değişken ataması;
# lst_salary_rank: medyanı kullanarak kategorik değer tabanını maaş_in_usd'de saklar
# lst_salaryrank: int değerini saklar (Yüksek:0 - Normal:1)
df['salary_in_usd'].describe()

mean_ = df['salary_in_usd'].mean()

lst_salary_rank = []
lst_salary_rank_ = []
for s in df.salary_in_usd:
    if(s < mean_):
        lst_salary_rank.append('Normal')
        lst_salary_rank_.append(0)
    if(s >= mean_):
        lst_salary_rank.append('High')
        lst_salary_rank_.append(1)

df['salary_rank'] = lst_salary_rank
df.head(30)



# --------------------------------------4.)Data Visualization-----------------------------------------

# 2020 den 2022 e veri bilimi alanındaki maaş ortalama artış trendlerini temsil eden grafik;
fig, axs = plt.subplots(1, 2, figsize=(15,8))
sns.barplot(ax = axs[0], data=df.groupby('work_year')['salary_in_usd'].mean().to_frame().reset_index(),
           x='work_year', y='salary_in_usd', palette='PuBuGn')

# Deneyim seviyesine göre maaş dağılımı;
df.groupby('experience_level')['salary_in_usd'].mean().sort_values().plot(kind='pie', cmap='GnBu');


# Maaşların kıtalara göre dağılımı;
plt.figure(figsize=(15,8))
sns.barplot(data=df.groupby('Continent')['salary_in_usd'].mean().sort_values().to_frame().reset_index(),
            x='Continent', y='salary_in_usd', palette='cividis');

#En yüksek maaş alan kişinin bilgisi;
df[df['salary_in_usd'] == df['salary_in_usd'].max()]

#En düşük maaş alan kişinin bilgisi;
df[df['salary_in_usd'] == df['salary_in_usd'].min()]

# Veri bilimi şirketlerinin çalıştığı lokasyonların dağılımı;
plt.figure(figsize=(15,8))
df.groupby('company_location')['company_size'].count().sort_values().plot(kind='bar', cmap='cividis');


# Çalışanların pozisyonlara göre dağılımı;
plt.figure(figsize=(15,8))
df.groupby('main_role')['main_role'].count().sort_values().plot(kind='bar', cmap='Set2')

# Pozisyona göre maaş dağılımı;
plt.figure(figsize=(15,8))
sns.barplot(data=df.groupby('main_role')['salary_in_usd'].mean().sort_values().to_frame().reset_index(),
            x='main_role', y='salary_in_usd', palette='plasma');

#Ticari sektörünün ve Yapay Zeka sektörünün maaşlara göre dağılımı;
plt.figure(figsize=(12,6))
df.groupby('main_field')['salary_in_usd'].mean().plot(kind='pie', cmap='Pastel1')


# Comapny size ile maaş arasındaki ilişki;
plt.figure(figsize=(12,8))
sns.barplot(data=df.groupby('company_size')['salary_in_usd'].mean().sort_values().to_frame().reset_index(),
            x='company_size', y='salary_in_usd', palette='plasma');

# Çalışma düzeninin maaşlara etkisi;
plt.figure(figsize=(12,6))
df.groupby('remote_ratio')['salary_in_usd'].mean().plot(kind='bar', cmap='Pastel1');
#

# Şirket lokasyonlarının maaşlara olan etkisi;
avg_sal_per_comp_location=df.groupby('company_location').mean()['salary_in_usd']

plt.figure(figsize=(20,20))
sns.barplot(x=avg_sal_per_comp_location.values,y=avg_sal_per_comp_location.index)
plt.ylabel('Company location')
plt.xlabel('Average Salary(USD)')
plt.title('Average Salary of Data Science jobs based on company location', y=1.01);

# Ülkelerin gelişmişlik seviyesinin maaşlara etkisi;
# ---grafik gösterge
plt.figure(figsize=(8,8))
sns.countplot(data=df, x='salary_rank', hue='Developed/Developing');



# ------------------------------------------5.) Hipotez Testleri------------------------------------

# Question : Company_size değişkeninin maaş bazında etkisi var mı ?

# 1.)Normallik varsayımına uyuyor mu ?

# -----Shapiro-----
# Ho:Normallik varsayamına uymuyor .
# H1:Normallik varsayamına uyuyor.
salary_comp = pd.DataFrame(columns= ['S', 'M', 'L'])

count = 0
for i in salary_comp:
    sample_ = df.salary_in_usd.loc[df['company_size'] == i]
    sample_ = np.array([np.mean(sample_.sample(20).values) for j in range(50)])
    salary_comp[i] = sample_
    count += 1

salary_comp.head()

# Test applied
alpha = .05
for i in salary_comp:
    shapiro, p = stats.shapiro(salary_comp[i])
    print(f'Critical value (Shapiro-based) ({i}) = {shapiro:4f}, p-value = {p:4f}')
    if (p < alpha):
        print('p-value < alpha red H0 ==> Normal değil!!!')
    else:
        print('Normallik testi başarılı!!!!')

# Anova test
# Company size değişkeninin maaşlar bazında farklılık oluşturur mu?
# Ho:Aralarında comapny size dan kaynaklı oluşan bir fark yoktur.
# H1:Aralarında comapny size dan kaynaklı oluşan bir fark vardır.
f, p = stats.f_oneway(salary_comp.S, salary_comp.M, salary_comp.L)

# conclusion base on p-value
print(f'p-value = {p:.2f}, alpha = {alpha:.2f}')
if (p < alpha):
    print('p-value < alpha red H0')
    print('Comapny_size değişkenleri arasında maaş bazında fark vardır')
    print('Tukey HSD is required')
else:
    print('Aralarında anlamlı bir fark yoktur.')


#  ---Tukay-HSD---
# Company_size değişkeni arasındaki maaş farklılığını ölçmek için tukey testi yapılır.

salary_comp['index'] = salary_comp.index
salary_comp_melt = pd.melt(salary_comp.reset_index(), id_vars = ['index'], value_vars = ['S', 'M', 'L'])
m_comp = pairwise_tukeyhsd(endog = salary_comp_melt['value'], groups = salary_comp_melt['variable'],
                                           alpha = 0.05)
print(m_comp)

# Grafiğe ve Hipotez testine göre,
#  L ve M şirket grubu hariç her şirket büyüklüğü arasında maaş miktarında farklılıklar olduğunu güvenle söyleyebiliriz. H0'ı reddedemeyen şirket grubu -> farklı değiller

# ----Chi Square Test----
#Question: ülkelerin gelişmişlik seviyesinin maaşlara etkisinin olup olmadığı gözlemleyiniz.

# H0: Gelişmekte olan ülke durumu ve maaş skalası bağımsızdır
# H1: Gelişmekte olan ülke statüsü ve maaş skalası bağımlıdır.

table_ = pd.crosstab(df['salary_rank'], df['Developed/Developing'])
table_

stat, p, dof, expected = chi2_contingency(table_)
print(f'Trị số p = {p:.2f}, alpha = {alpha:.2f}')

if (p < alpha):
    print('p-value < alpha red H0 ==> Gelişmekte olan ülke durumu ve maaş skalası bağımlıdır')
else:
    print('Gelişmekte olan ülke statüsü ve maaş skalası bağımsızdır')
#
# Trị số p = 0.00, alpha = 0.05
# p-value < alpha reject H0 ==> Gelişmekte olan ülke durumu ve maaş skalası bağımlıdır
# Gelişmekte olan ülke durumu ve maaş skalası bağımlıdır


# ----------------------Some Exercises-------------------
# Not : Hipotez ve görsel işlemleri için gerekli olan ama model sırasında ihtiyacımız olmayan bazı değişiklikler bu bölümde yapılmıştır.

# Employee_residence ve company_location değişkenleri feature engineering bölümünde bir başka değişkene dönüştürüldüğü için drop edilir.
df.drop(["employee_residence",'company_location'], axis=1, inplace=True)
df.columns
df.head()


# company size adında bir fonksiyon ile şirket büyüklüğüne göre skorlar verildi.
def Company_size(str):
    if str == "L":
        return 3
    elif str == "M":
        return 2
    else:
        return 1
# şirket büyüklüğüne göre yeni değişken oluşturulup, atama yapıldı.
df["Com_size"] = df["company_size"].apply( lambda x : Company_size(x))

#exper_level adında fonksiyon oluşturuldu ve skorlar verildi.
def exper_level (str):
    if str == "EN":
        return 1
    elif str == "MI":
        return 2
    elif str == "SE":
        return 3
    else:
        return 4

# exp_level adında yeni değişken oluşturularak skorlar atandı.
df["exp_level"] = df["experience_level"].apply( lambda x : exper_level(x))


# dev_stat adında fonk kurularak skorlar belirlendi.
def dev_stat(str):
    if str == "developed":
        return 1
    else:
        return 0

# dev_stat adında yeni değişken oluşturularak skorlar atandı.
df["dev_stat"] = df["Developed/Developing"].apply( lambda x : dev_stat(x))



#bu değişkenlere artık ihtiyacımız yok yenisi olduğu için drop edebiliriz.
df.drop(["experience_level","company_size","Developed/Developing"], axis=1,inplace=True)


#bağımlı değişkeni, one -hot encoding ten geçirmek istemediğimiz için çıkarıyoruz.
data_one_hot=[col for col in df.columns if col not in ["salary_in_usd"]]

#label_encoding amacıyla yaptığımız yeni değişken atamalarını da one_hot datasından çıkarıyoruz.
data_one_hot=[col for col in data_one_hot if col not in [ 'Com_size','exp_level','dev_stat']]

df.columns
df.head()


# -------------------------------------------Standartlaştırma--------------------------------------
# KATEGORİK DEĞİŞKEN STANDARTLAŞTIRMA
df = pd.get_dummies(df,columns=data_one_hot,drop_first=True)
df.head()
df.columns

#NÜMERİK DEĞİŞKEN STANDARTLAŞTIRMA
num_cols
# Out[15]: ['Unnamed: 0', 'salary', 'salary_in_usd']  # nümerik değğişkenler arasında sadece hedef değişken oldğu için nümerik değişkenler için standartlaştırma yapılmaz.

# ----------------------------------------Modelling--------------------------------
# define dataset
X, y = df.drop("salary_in_usd",axis=1) , df["salary_in_usd"]

# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


models = {
    "Linear regression":LinearRegression(),
    "Lasso ":LassoCV(),
    "Ridge":RidgeCV(),
    "ElasticNet":ElasticNetCV(),
    "LightGBM": LGBMRegressor(),
    'KNN': KNeighborsRegressor(),
    'RF': RandomForestRegressor()
}
Results = {
    "Model":[],
    "Train Score":[],
    "Test Score":[],
    "RMSE":[]
}

for name, model in models.items():
    model.fit(X_train,np.log(y_train))
    train_s = model.score(X_train,np.log(y_train))
    test_s = model.score(X_test,np.log(y_test))
    y_pred = model.predict(X_test)
    RMSE = mean_squared_error((y_pred),np.log(y_test))
    Results["Model"].append(name)
    Results["Train Score"].append(train_s)
    Results["Test Score"].append(test_s)
    Results["RMSE"].append(RMSE)
    print("Model: " , name)
    print("Train Score: " , train_s)
    print("Test Score : " , test_s)
    print("RMSE : " , round(RMSE,2))
    print("===========================")


# ----OUTPUT----
# Model:  Linear regression
# Train Score:  0.6995738981811872
# Test Score :  0.6820749260163337
# RMSE :  0.23
# ===========================
# Model:  Lasso
# Train Score:  0.6721848787348771
# Test Score :  0.6714809865323834
# RMSE :  0.24
# ===========================
# Model:  Ridge
# Train Score:  0.671148782840385
# Test Score :  0.6853750284390622
# RMSE :  0.23
# ===========================
# Model:  ElasticNet
# Train Score:  0.6709815236576981
# Test Score :  0.6761676978453728
# RMSE :  0.23
# ===========================
# Model:  LightGBM
# Train Score:  0.7635170159875003
# Test Score :  0.6829142469987826
# RMSE :  0.23
# ===========================
# Model:  KNN
# Train Score:  0.7023270481161035
# Test Score :  0.6513750186746945
# RMSE :  0.25
# ===========================
# Model:  RF
# Train Score:  0.9035081705976358
# Test Score :  0.6742641061352503
# RMSE :  0.23
# ===========================




# ----------------------Conclusion//NOTLAR----------------

# Veri bilimi işleri daha popüler hale geliyor.
# Bir çalışan mümkün olan en yüksek maaşı almak istiyorsa, tercihi Amerika Birleşik Devletleri olmalıdır. Ama tabii ki bu analiz tamamen maaşlara dayalı. Buna cevap vermek için, yaşam maliyetini, sağlık bakımını vb. de analiz etmeliyiz.
# Büyük ve orta ölçekli şirketler, küçük şirketlere göre daha fazla maaş veriyor.
# Remote(home office) daha popüler. Bir başka çıkarım ile de remote çalışanların daha yüksek maaş aldığı gözlemlenir.
# Üst düzey bir deneyim kazandıktan sonra maaş artışı şiddetli hale gelir.
# Veri Mühendisleri, Veri Bilimcileri ve Makine Öğrenimi Mühendisleri en değerli unvanlardır (ortalama maaşlarına bakarak).
# Verilerin çoğunluğu Amerika Birleşik Devletleri'ne dayanmaktadır. Amerika Birleşik Devletleri diğer ülkelerden çok daha yüksek ücretler ödediğinden, bu ortalama maaşların gerçeği yansıttığını düşünmüyorum. Sonuç olarak, bu ortalama maaşlara bakarak beklentilerimi temellendirmem.
# Bu analizi yaptıktan sonra, veri bilimi ile ilgili bir iş aramanın maaş ve remote çalışma açısından iyi bir kariyer seçimi olduğuna düşünülebilir.






# ----------------OUTPUT------
# ##################### Shape #####################
# (607, 12)
# ##################### Types #####################
# Unnamed: 0             int64
# work_year              int64
# experience_level      object
# employment_type       object
# job_title             object
# salary                 int64
# salary_currency       object
# salary_in_usd          int64
# employee_residence    object
# remote_ratio           int64
# company_location      object
# company_size          object
# dtype: object
# ##################### Head #####################
#    Unnamed: 0  work_year experience_level employment_type                   job_title  salary salary_currency  salary_in_usd employee_residence  remote_ratio company_location company_size
# 0           0       2020               MI              FT              Data Scientist   70000             EUR          79833                 DE             0               DE            L
# 1           1       2020               SE              FT  Machine Learning Scientist  260000             USD         260000                 JP             0               JP            S
# 2           2       2020               SE              FT           Big Data Engineer   85000             GBP         109024                 GB            50               GB            M
# 3           3       2020               MI              FT        Product Data Analyst   20000             USD          20000                 HN             0               HN            S
# 4           4       2020               SE              FT   Machine Learning Engineer  150000             USD         150000                 US            50               US            L
# ##################### Tail #####################
#      Unnamed: 0  work_year experience_level employment_type      job_title  salary salary_currency  salary_in_usd employee_residence  remote_ratio company_location company_size
# 602         602       2022               SE              FT  Data Engineer  154000             USD         154000                 US           100               US            M
# 603         603       2022               SE              FT  Data Engineer  126000             USD         126000                 US           100               US            M
# 604         604       2022               SE              FT   Data Analyst  129000             USD         129000                 US             0               US            M
# 605         605       2022               SE              FT   Data Analyst  150000             USD         150000                 US           100               US            M
# 606         606       2022               MI              FT   AI Scientist  200000             USD         200000                 IN           100               US            L
# ##################### NA #####################
# Unnamed: 0            0
# work_year             0
# experience_level      0
# employment_type       0
# job_title             0
# salary                0
# salary_currency       0
# salary_in_usd         0
# employee_residence    0
# remote_ratio          0
# company_location      0
# company_size          0
# dtype: int64
# ##################### Quantiles #####################
#                  0.000     0.050      0.500      0.950       0.990        1.000
# Unnamed: 0       0.000    30.300    303.000    575.700     599.940      606.000
# work_year     2020.000  2020.000   2022.000   2022.000    2022.000     2022.000
# salary        4000.000 30000.000 115000.000 450000.000 5934000.000 30400000.000
# salary_in_usd 2859.000 20000.000 101570.000 220110.000  403500.000   600000.000
# remote_ratio     0.000     0.000    100.000    100.000     100.000      100.000
# ##################### Describe #####################
#                 count       mean         std      min       25%        50%        75%          max
# Unnamed: 0    607.000    303.000     175.370    0.000   151.500    303.000    454.500      606.000
# work_year     607.000   2021.405       0.692 2020.000  2021.000   2022.000   2022.000     2022.000
# salary        607.000 324000.063 1544357.487 4000.000 70000.000 115000.000 165000.000 30400000.000
# salary_in_usd 607.000 112297.870   70957.259 2859.000 62726.000 101570.000 150000.000   600000.000
# remote_ratio  607.000     70.923      40.709    0.000    50.000    100.000    100.000      100.000
# Observations: 607
# Variables: 12
# cat_cols: 6
# num_cols: 3
# cat_but_car: 3
# num_but_cat: 2
# cat_cols: ['experience_level', 'employment_type', 'salary_currency', 'company_size', 'work_year', 'remote_ratio']
# num_cols: ['Unnamed: 0', 'salary', 'salary_in_usd']
# cat_but_car: ['job_title', 'employee_residence', 'company_location']
# num_but_cat: ['work_year', 'remote_ratio']
#     experience_level  Ratio
# SE               280 46.129
# MI               213 35.091
# EN                88 14.498
# EX                26  4.283
# ##########################################
#     employment_type  Ratio
# FT              588 96.870
# PT               10  1.647
# CT                5  0.824
# FL                4  0.659
# ##########################################
#      salary_currency  Ratio
# USD              398 65.568
# EUR               95 15.651
# GBP               44  7.249
# INR               27  4.448
# CAD               18  2.965
# JPY                3  0.494
# PLN                3  0.494
# TRY                3  0.494
# CNY                2  0.329
# MXN                2  0.329
# HUF                2  0.329
# DKK                2  0.329
# SGD                2  0.329
# BRL                2  0.329
# AUD                2  0.329
# CLP                1  0.165
# CHF                1  0.165
# ##########################################
#    company_size  Ratio
# M           326 53.707
# L           198 32.619
# S            83 13.674
# ##########################################
#       work_year  Ratio
# 2022        318 52.389
# 2021        217 35.750
# 2020         72 11.862
# ##########################################
#      remote_ratio  Ratio
# 100           381 62.768
# 0             127 20.923
# 50             99 16.310
# ##########################################
# count   607.000
# mean    303.000
# std     175.370
# min       0.000
# 5%       30.300
# 10%      60.600
# 20%     121.200
# 30%     181.800
# 40%     242.400
# 50%     303.000
# 60%     363.600
# 70%     424.200
# 80%     484.800
# 90%     545.400
# 95%     575.700
# 99%     599.940
# max     606.000
# Name: Unnamed: 0, dtype: float64
# count        607.000
# mean      324000.063
# std      1544357.487
# min         4000.000
# 5%         30000.000
# 10%        42720.000
# 20%        60000.000
# 30%        80000.000
# 40%        99020.000
# 50%       115000.000
# 60%       133928.000
# 70%       152600.000
# 80%       180000.000
# 90%       237000.000
# 95%       450000.000
# 99%      5934000.000
# max     30400000.000
# Name: salary, dtype: float64
# count      607.000
# mean    112297.870
# std      70957.259
# min       2859.000
# 5%       20000.000
# 10%      33689.200
# 20%      54957.000
# 30%      71337.600
# 40%      87932.000
# 50%     101570.000
# 60%     120000.000
# 70%     140000.000
# 80%     160000.000
# 90%     200000.000
# 95%     220110.000
# 99%     403500.000
# max     600000.000
# Name: salary_in_usd, dtype: float64
# salary_in_usd
# Observations: 607
# Variables: 9
# cat_cols: 5
# num_cols: 1
# cat_but_car: 3
# num_but_cat: 0
# cat_cols: ['work_year', 'experience_level', 'employment_type', 'remote_ratio', 'company_size']
# num_cols: ['salary_in_usd']
# cat_but_car: ['job_title', 'employee_residence', 'company_location']
# num_but_cat: []
# Critical value (Shapiro-based) (S) = 0.960338, p-value = 0.091836
# Normallik testi başarılı!!!!
# Critical value (Shapiro-based) (M) = 0.984781, p-value = 0.762162
# Normallik testi başarılı!!!!
# Critical value (Shapiro-based) (L) = 0.932239, p-value = 0.006723
# p-value < alpha red H0 ==> Normal değil!!!
# p-value = 0.00, alpha = 0.05
# p-value < alpha red H0
# Comapny_size değişkenleri arasında maaş bazında fark vardır
# Tukey HSD is required
#      Multiple Comparison of Means - Tukey HSD, FWER=0.05
# ==============================================================
# group1 group2  meandiff  p-adj     lower       upper    reject
# --------------------------------------------------------------
#      L      M  -3587.984 0.4886 -11012.6738   3836.7058  False
#      L      S -39149.173    0.0 -46573.8628 -31724.4832   True
#      M      S -35561.189    0.0 -42985.8788 -28136.4992   True
# --------------------------------------------------------------
# Trị số p = 0.00, alpha = 0.05
# p-value < alpha red H0 ==> Gelişmekte olan ülke durumu ve maaş skalası bağımlıdır
# Model:  Linear regression
# Train Score:  0.6995738981811872
# Test Score :  0.6820749260163337
# RMSE :  0.23
# ===========================
# Model:  Lasso
# Train Score:  0.6721848787348771
# Test Score :  0.6714809865323834
# RMSE :  0.24
# ===========================
# Model:  Ridge
# Train Score:  0.671148782840385
# Test Score :  0.6853750284390622
# RMSE :  0.23
# ===========================
# Model:  ElasticNet
# Train Score:  0.6709815236576981
# Test Score :  0.6761676978453728
# RMSE :  0.23
# ===========================
# Model:  LightGBM
# Train Score:  0.7635170159875003
# Test Score :  0.6829142469987826
# RMSE :  0.23
# ===========================
# Model:  KNN
# Train Score:  0.7023270481161035
# Test Score :  0.6513750186746945
# RMSE :  0.25
# ===========================
# Model:  RF
# Train Score:  0.9035081705976358
# Test Score :  0.6742641061352503
# RMSE :  0.23
# ===========================
