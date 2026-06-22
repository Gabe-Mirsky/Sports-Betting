"""URL routes for the predictions app."""

from django.urls import path

from . import views

app_name = "predictions"

urlpatterns = [
    path("", views.home, name="home"),
    path("predictions/", views.prediction_list, name="list"),
    path("predictions/<int:pk>/", views.prediction_detail, name="detail"),
    path("leaderboard/", views.leaderboard, name="leaderboard"),
    path("parlay/", views.parlay_creator, name="parlay_creator"),
    path("parlay/<int:pk>/", views.parlay_detail, name="parlay_detail"),
    path("recommendations/", views.recommendations, name="recommendations"),
    path("how-it-works/", views.how_it_works, name="how_it_works"),
    path("add/", views.add_prediction, name="add"),
    path("add/preview/", views.preview_prediction, name="preview"),
    path("about/", views.about, name="about"),
]
