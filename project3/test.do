cd "C:\Users\JBH\Dropbox\10_semester\Advanced Microeconometrics\project3"
xtset ma
use "test.dta",clear

clogit s logp home logp_x_home cy hp we li he MCC VW alfa_romeo audi citroen daewoo daf fiat ford honda hyundai innocenti lancia mazda mercedes mitsubishi nissan opel peugeot renault rover saab seat skoda suzuki tal_hillman tal_matra tal_simca tal_sunb talbot toyota volvo, group(ye)