"""Django admin registration so the owner can approve/edit predictions."""

from __future__ import annotations

from django.contrib import admin

from .models import MatchupPrediction, Parlay, ParlayLeg


@admin.register(MatchupPrediction)
class MatchupPredictionAdmin(admin.ModelAdmin):
    list_display = (
        "matchup",
        "sport",
        "league",
        "game_date",
        "predicted_outcome",
        "confidence_level",
        "status",
    )
    list_filter = ("status", "sport", "league", "confidence_level", "data_quality")
    search_fields = ("team_a", "team_b", "fixture_id", "predicted_outcome")
    list_editable = ("status",)
    date_hierarchy = "game_date"
    actions = ("approve_predictions", "hide_predictions")

    @admin.action(description="Approve selected predictions (make public)")
    def approve_predictions(self, request, queryset):
        updated = queryset.update(status=MatchupPrediction.STATUS_APPROVED)
        self.message_user(request, f"{updated} prediction(s) approved.")

    @admin.action(description="Hide selected predictions")
    def hide_predictions(self, request, queryset):
        updated = queryset.update(status=MatchupPrediction.STATUS_HIDDEN)
        self.message_user(request, f"{updated} prediction(s) hidden.")


class ParlayLegInline(admin.TabularInline):
    model = ParlayLeg
    extra = 0


@admin.register(Parlay)
class ParlayAdmin(admin.ModelAdmin):
    list_display = ("__str__", "leg_count", "combined_probability_pct", "created_at")
    inlines = (ParlayLegInline,)
    search_fields = ("name", "notes")
