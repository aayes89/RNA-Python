//importo las clases a usar
import numpy
import pandas
import matplotlib.pyplot as plt
import scipy.stats
plt.style.use("bmh")
%config InlineBackend.figure_formats=["png"]

//importo los datos a leer
data = pandas.read_csv("combined_cycle_power_plant.csv") 
data.head()
data.describe()

//Predicción (entrenamiento)
import statsmodels.formula.api as smf
lm = smf.ols(formula="PE ~ AT + V + AP + RH", data=data).fit()
lm.params
lm.predict(pandas.DataFrame({"AT":[9.48], "V":[44.71], "AP": [1019.12]. "RH":[66.43]}))
//esa fue una predicción para una salida de 478 MW aproximadamente
//calculo el residuo de la predicción para indicar que el modelo lineal no sea el apropiado en cada caso
residuo = lm.predict(data) - data.PE
//lo grafico 
scipy.stats.probplot(residuo, dist=scipy.stats.norm, plot=plt.figure().add_subplot(111));
//veo los resultados finales
lm.summary()
