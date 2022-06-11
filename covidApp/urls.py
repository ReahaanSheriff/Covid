from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home,name='home'),
    path('deleteFile<str:fileName>/',views.deleteFile,name='deleteFile'),
    path('data/<str:fileCSV>/',views.datatable,name='datatable'),
    path('displayGraph/',views.displayGraph,name='displayGraph'),
    path('models/<str:temp>/',views.models,name='models'),
    path('selectFile/<str:f>/',views.selectFile,name='selectFile'),
    path('metrics/',views.metrics,name='metrics')
]
