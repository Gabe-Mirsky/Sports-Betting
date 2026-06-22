"""Forms for adding predictions and saving parlays."""

from __future__ import annotations

from django import forms

from .models import MatchupPrediction, Parlay


class AddPredictionForm(forms.ModelForm):
    """Collects a raw matchup; reasoning is filled in by the AI service."""

    class Meta:
        model = MatchupPrediction
        fields = [
            "sport",
            "league",
            "game_date",
            "team_a",
            "team_b",
            "prob_team_a_win",
            "prob_draw",
            "prob_team_b_win",
            "predicted_outcome",
        ]
        widgets = {
            "game_date": forms.DateTimeInput(
                attrs={"type": "datetime-local"}, format="%Y-%m-%dT%H:%M"
            ),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["game_date"].input_formats = ["%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S"]
        self.fields["predicted_outcome"].required = False
        for prob in ("prob_team_a_win", "prob_draw", "prob_team_b_win"):
            self.fields[prob].required = False
            self.fields[prob].help_text = "0-1 probability"

    def clean(self):
        cleaned = super().clean()
        # Derive the predicted outcome from probabilities when not supplied.
        if not cleaned.get("predicted_outcome"):
            options = {
                f"{cleaned.get('team_a', 'Team A')} win": cleaned.get("prob_team_a_win") or 0,
                "Draw": cleaned.get("prob_draw") or 0,
                f"{cleaned.get('team_b', 'Team B')} win": cleaned.get("prob_team_b_win") or 0,
            }
            cleaned["predicted_outcome"] = max(options, key=options.get)
        return cleaned


class ParlayForm(forms.ModelForm):
    """Name/notes for a saved parlay (legs handled in the view)."""

    class Meta:
        model = Parlay
        fields = ["name", "notes"]
        widgets = {
            "name": forms.TextInput(attrs={"placeholder": "Optional parlay name"}),
            "notes": forms.Textarea(attrs={"rows": 2, "placeholder": "Optional notes"}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["name"].required = False
        self.fields["notes"].required = False
