import unittest
import arima_model

class Testarima(unittest.TestCase):
    def test_arimaForecasting(self):
        second = "G:/SEMESTER_6/covidForcastingProject/covid19Prediction/my_folder/images/arpredict.png"
        self.assertListEqual(arima_model.arimaForecasting('WHO-COVID-19-global-data.csv'),
         [second,9958.253824148016,99166819.22615857,9639.662747915643])