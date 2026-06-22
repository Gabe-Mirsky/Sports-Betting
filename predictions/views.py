"""Views for the prediction website."""

from __future__ import annotations

from django.contrib import messages
from django.db.models import Q
from django.shortcuts import get_object_or_404, redirect, render

from .feature_catalog import FEATURE_GROUPS, feature_lookup
from .forms import AddPredictionForm, ParlayForm
from .models import MatchupPrediction, Parlay, ParlayLeg
from .services.ai_reasoning import define_prediction_reasoning
from .services.model_explainer import get_model_explanation
from .services.parlay_recommender import recommend_parlays

_CONF_THRESHOLD = {"1": "Low", "2": "Medium", "3": "High"}
_CONF_MIN_RANK = {"1": 1, "2": 2, "3": 3}


def _approved():
    return MatchupPrediction.objects.filter(status=MatchupPrediction.STATUS_APPROVED)


def _apply_filters(queryset, request):
    """Apply sport/league/confidence/search GET filters and return (qs, context)."""

    sport = request.GET.get("sport", "").strip()
    league = request.GET.get("league", "").strip()
    confidence = request.GET.get("confidence", "").strip()
    query = request.GET.get("q", "").strip()

    if sport:
        queryset = queryset.filter(sport=sport)
    if league:
        queryset = queryset.filter(league=league)
    if query:
        queryset = queryset.filter(Q(team_a__icontains=query) | Q(team_b__icontains=query))

    rows = list(queryset)
    if confidence in _CONF_MIN_RANK:
        min_rank = _CONF_MIN_RANK[confidence]
        rows = [row for row in rows if row.confidence_rank >= min_rank]

    context = {
        "sports": sorted(_approved().exclude(sport="").values_list("sport", flat=True).distinct()),
        "leagues": sorted(_approved().exclude(league="").values_list("league", flat=True).distinct()),
        "selected_sport": sport,
        "selected_league": league,
        "selected_confidence": confidence,
        "query": query,
    }
    return rows, context


def home(request):
    approved = _approved()
    top = sorted(
        approved,
        key=lambda p: (p.confidence_score or 0, p.pick_probability or 0),
        reverse=True,
    )[:6]
    context = {
        "prediction_count": approved.count(),
        "sport_count": approved.exclude(sport="").values("sport").distinct().count(),
        "league_count": approved.exclude(league="").values("league").distinct().count(),
        "parlay_count": Parlay.objects.count(),
        "top_predictions": top,
    }
    return render(request, "predictions/home.html", context)


def prediction_list(request):
    rows, context = _apply_filters(_approved(), request)
    context["predictions"] = rows
    return render(request, "predictions/prediction_list.html", context)


def prediction_detail(request, pk):
    # Show approved or pending rows (pending get a banner); hide "hidden" ones.
    prediction = get_object_or_404(
        MatchupPrediction.objects.exclude(status=MatchupPrediction.STATUS_HIDDEN), pk=pk
    )
    return render(request, "predictions/prediction_detail.html", {"prediction": prediction})


def leaderboard(request):
    sort = request.GET.get("sort", "confidence")
    rows = list(_approved())
    if sort == "probability":
        rows.sort(key=lambda p: (p.pick_probability or 0), reverse=True)
    else:
        sort = "confidence"
        rows.sort(key=lambda p: (p.confidence_score or 0, p.pick_probability or 0), reverse=True)
    return render(
        request,
        "predictions/leaderboard.html",
        {"predictions": rows, "sort": sort},
    )


def parlay_creator(request):
    if request.method == "POST":
        leg_ids = request.POST.getlist("legs")
        selected = list(_approved().filter(pk__in=leg_ids))
        if len(selected) < 2:
            messages.error(request, "Pick at least two predictions to build a parlay.")
            return redirect("predictions:parlay_creator")

        form = ParlayForm(request.POST)
        if form.is_valid():
            parlay = form.save()
            for prediction in selected:
                ParlayLeg.objects.create(
                    parlay=parlay,
                    prediction=prediction,
                    pick_label=prediction.predicted_outcome or prediction.matchup,
                    match_label=prediction.matchup,
                    pick_probability=prediction.pick_probability,
                )
            messages.success(request, f"Saved a {parlay.leg_count}-leg parlay.")
            return redirect(parlay.get_absolute_url())

    rows, context = _apply_filters(_approved(), request)
    context.update(
        {
            "predictions": rows,
            "form": ParlayForm(),
            "saved_parlays": Parlay.objects.all()[:10],
        }
    )
    return render(request, "predictions/parlay_creator.html", context)


def parlay_detail(request, pk):
    parlay = get_object_or_404(Parlay, pk=pk)
    return render(request, "predictions/parlay_detail.html", {"parlay": parlay})


def recommendations(request):
    """Model-suggested parlays with plain-language reasoning."""
    recs = recommend_parlays(list(_approved()))
    return render(request, "predictions/recommendations.html", {"recommendations": recs})


def how_it_works(request):
    """Explain, in plain language, every variable and how probability is computed."""
    explanation = get_model_explanation()
    lookup = feature_lookup()

    sport_tables = []
    if explanation:
        for sport, info in explanation.get("sports", {}).items():
            weights = info.get("weights", {})
            # Main table: the curated, easy-to-read variables only.
            rows = [
                {"name": lookup[code]["name"], "desc": lookup[code]["desc"],
                 "group": lookup[code]["group"], "weight": weight}
                for code, weight in weights.items()
                if code in lookup
            ]
            rows.sort(key=lambda r: abs(r["weight"]), reverse=True)
            # "Kind of game" expands into many one-hot columns; show only the
            # handful that move the prediction the most, so the page stays simple.
            game_types = [
                {"name": code.replace("competition_type_", ""), "weight": weight}
                for code, weight in weights.items()
                if code.startswith("competition_type_")
            ]
            game_types.sort(key=lambda r: abs(r["weight"]), reverse=True)
            sport_tables.append({
                "sport": sport,
                "n_train": info.get("n_train"),
                "intercept": info.get("intercept_toward_team_a"),
                "rows": rows,
                "game_types": game_types[:8],
            })

    context = {
        "groups": FEATURE_GROUPS,
        "explanation": explanation,
        "sport_tables": sport_tables,
    }
    return render(request, "predictions/how_it_works.html", context)


def add_prediction(request):
    """Step 1: collect a raw matchup, generate AI reasoning, stash for preview."""

    if request.method == "POST":
        form = AddPredictionForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            context = {
                "sport": data.get("sport"),
                "league": data.get("league"),
                "team_a": data.get("team_a"),
                "team_b": data.get("team_b"),
                "prob_team_a_win": data.get("prob_team_a_win"),
                "prob_draw": data.get("prob_draw"),
                "prob_team_b_win": data.get("prob_team_b_win"),
                "predicted_outcome": data.get("predicted_outcome"),
            }
            reasoning = define_prediction_reasoning(context)
            pending = {
                "sport": data.get("sport", ""),
                "league": data.get("league", ""),
                "game_date": data["game_date"].isoformat() if data.get("game_date") else "",
                "team_a": data.get("team_a", ""),
                "team_b": data.get("team_b", ""),
                "prob_team_a_win": data.get("prob_team_a_win"),
                "prob_draw": data.get("prob_draw"),
                "prob_team_b_win": data.get("prob_team_b_win"),
                "predicted_outcome": data.get("predicted_outcome", ""),
                "key_reasons": "; ".join(reasoning.get("key_reasons", [])),
                "main_risks": "; ".join(reasoning.get("main_risks", [])),
                "ai_reasoning": reasoning.get("narrative", ""),
                "ai_provider": reasoning.get("provider", ""),
            }
            request.session["pending_prediction"] = pending
            return redirect("predictions:preview")
    else:
        form = AddPredictionForm()
    return render(request, "predictions/add_prediction.html", {"form": form})


def preview_prediction(request):
    """Step 2: preview the AI-defined prediction, then confirm to save it."""

    pending = request.session.get("pending_prediction")
    if not pending:
        messages.info(request, "Start by adding a matchup.")
        return redirect("predictions:add")

    if request.method == "POST":
        from datetime import datetime

        game_date = None
        if pending.get("game_date"):
            try:
                game_date = datetime.fromisoformat(pending["game_date"])
            except ValueError:
                game_date = None
        team_a = pending.get("team_a", "")
        team_b = pending.get("team_b", "")
        fixture_id = f"manual_{team_a}_{team_b}_{pending.get('game_date', '')}".lower().replace(" ", "_")
        prediction, _ = MatchupPrediction.objects.update_or_create(
            fixture_id=fixture_id,
            defaults={
                "sport": pending.get("sport", ""),
                "league": pending.get("league", ""),
                "game_date": game_date,
                "team_a": team_a,
                "team_b": team_b,
                "prob_team_a_win": pending.get("prob_team_a_win"),
                "prob_draw": pending.get("prob_draw"),
                "prob_team_b_win": pending.get("prob_team_b_win"),
                "predicted_outcome": pending.get("predicted_outcome", ""),
                "key_reasons": pending.get("key_reasons", ""),
                "main_risks": pending.get("main_risks", ""),
                "ai_reasoning": pending.get("ai_reasoning", ""),
                "model_version": "manual_ai_defined",
                # New manual entries wait for owner approval in the admin.
                "status": MatchupPrediction.STATUS_PENDING,
            },
        )
        del request.session["pending_prediction"]
        messages.success(
            request,
            "Prediction submitted. It is pending review and will appear once approved in the admin.",
        )
        return redirect(prediction.get_absolute_url())

    return render(request, "predictions/preview_prediction.html", {"pending": pending})


def about(request):
    return render(request, "predictions/about.html")
