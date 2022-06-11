import unittest
import prophet_model

class Testprophet(unittest.TestCase):
    def test_prophetForecasting(self):
        first = "G:/SEMESTER_6/covidForcastingProject/covid19Prediction/my_folder/images/protrain.png"
        second = "G:/SEMESTER_6/covidForcastingProject/covid19Prediction/my_folder/images/propredict.png"
        self.assertListEqual(prophet_model.prophetForecasting('WHO-COVID-19-global-data.csv'),
         [first,second,55397.00280770074,7929814506.702507,89049.50593182708])