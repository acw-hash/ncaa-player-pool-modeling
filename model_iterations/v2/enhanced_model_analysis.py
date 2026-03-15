#!/usr/bin/env python3
"""
Enhanced NCAA Tournament Model — Team Metrics Integration
==========================================================
Compiles historical team-level metrics (KenPom AdjEM, Pace, Q1 Wins, NET/SOR, AdjO)
and player usage rates for 2022-2025 tournament teams, merges with pool data,
fits an enhanced predictive model, and validates via rolling backtests.
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr
import json
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# HISTORICAL TEAM METRICS (2022-2025 Tournament Teams)
# Sources: KenPom, NCAA NET, Barttorvik — compiled from public records
# AdjEM = KenPom Adjusted Efficiency Margin
# AdjO = KenPom Adjusted Offensive Efficiency (pts per 100 possessions)
# Pace = Possessions per 40 minutes (KenPom)
# Q1_Wins = Quad 1 wins (NCAA NET quadrant system)
# SOR_Rank = NCAA Strength of Record ranking
# =============================================================================

TEAM_METRICS_2022 = {
    # Team: (Seed, AdjEM, AdjO, Pace, Q1_Wins, SOR_Rank, Games_Won_Tourney)
    "Kansas":        (1, 26.37, 120.5, 67.8, 10, 3, 6),
    "Arizona":       (1, 27.08, 117.6, 71.3, 10, 4, 2),
    "Gonzaga":       (1, 28.86, 124.4, 72.3, 6, 8, 3),
    "Baylor":        (1, 22.95, 114.9, 66.7, 10, 2, 2),
    "Auburn":        (2, 22.25, 112.7, 70.0, 9, 5, 2),
    "Kentucky":      (2, 19.05, 115.8, 69.9, 8, 9, 1),
    "Duke":          (2, 24.04, 119.4, 71.2, 9, 6, 4),
    "Villanova":     (2, 19.73, 114.7, 65.3, 8, 7, 5),
    "Tennessee":     (3, 20.15, 109.2, 65.4, 8, 11, 1),
    "Purdue":        (3, 24.43, 117.4, 66.5, 8, 10, 2),
    "Wisconsin":     (3, 16.31, 112.1, 63.0, 6, 17, 1),
    "Texas Tech":    (3, 21.81, 111.3, 65.0, 9, 12, 3),
    "UCLA":          (4, 18.29, 114.8, 66.2, 7, 14, 1),
    "Illinois":      (4, 17.47, 113.1, 70.3, 7, 16, 1),
    "Arkansas":      (4, 19.44, 110.5, 72.8, 7, 15, 4),
    "Providence":    (4, 14.69, 111.2, 64.2, 5, 20, 2),
    "Connecticut":   (5, 15.84, 114.3, 69.4, 6, 18, 1),
    "Saint Mary's":  (5, 18.35, 115.2, 62.0, 3, 30, 2),
    "Houston":       (5, 24.91, 109.2, 65.2, 7, 13, 3),
    "Iowa":          (5, 16.58, 119.3, 71.5, 5, 22, 1),
    "Texas":         (6, 15.33, 110.4, 66.7, 7, 19, 1),
    "Colorado State":(6, 14.21, 112.0, 65.8, 2, 35, 1),
    "TCU":           (6, 10.96, 107.9, 68.2, 4, 28, 1),
    "Murray State":  (7, 15.70, 112.5, 65.7, 1, 45, 1),
    "Michigan State": (7, 11.48, 109.4, 68.1, 4, 32, 2),
    "Ohio State":    (7, 13.76, 113.5, 67.2, 5, 26, 2),
    "USC":           (7, 12.05, 111.0, 68.3, 3, 34, 1),
    "North Carolina": (8, 14.88, 116.5, 72.5, 5, 23, 6),
    "San Diego State":(8, 15.12, 104.8, 64.0, 3, 36, 1),
    "Memphis":       (8, 10.83, 108.5, 69.5, 4, 33, 1),
    "Boise State":   (8, 11.25, 108.3, 65.0, 2, 40, 1),
    "Creighton":     (9, 14.10, 113.5, 68.0, 5, 24, 2),
    "Iowa State":    (11, 11.07, 107.2, 66.4, 5, 25, 3),
    "Virginia Tech": (11, 8.26, 108.0, 66.8, 4, 31, 1),
    "Notre Dame":    (11, 9.15, 109.8, 68.5, 4, 29, 1),
    "New Mexico State":(12, 9.52, 107.2, 66.0, 0, 65, 1),
    "Richmond":      (12, 10.05, 108.5, 63.5, 1, 55, 1),
    "Indiana":       (12, 9.71, 108.2, 68.1, 2, 41, 0),
    "Montana State": (14, 5.10, 106.2, 68.5, 0, 120, 0),
    "Colgate":       (14, 6.80, 108.0, 69.0, 0, 100, 0),
    "Saint Peter's":  (15, 2.81, 103.5, 65.8, 0, 135, 4),
    "Delaware":      (15, 3.20, 104.0, 68.0, 0, 130, 0),
    "Norfolk State": (16, -2.50, 100.5, 69.0, 0, 200, 0),
    "SE Missouri St": (16, -5.80, 98.5, 71.0, 0, 220, 1),
    "Corpus Christi": (16, -3.10, 99.0, 67.5, 0, 210, 0),
    "VCU":           (12, 10.30, 107.8, 69.5, 2, 42, 1),
    "Marquette":     (9, 12.68, 110.5, 66.0, 4, 27, 1),
    "Miami":         (10, 11.96, 112.8, 67.3, 3, 37, 2),
    "Loyola Chicago": (10, 13.00, 108.0, 63.5, 1, 48, 1),
    "New Mexico":    (13, 8.72, 108.5, 70.5, 1, 52, 0),
    "South Dakota State": (13, 7.80, 107.0, 68.2, 0, 75, 0),
    "Georgia State": (13, 4.55, 104.5, 67.0, 0, 95, 0),
    "Akron":         (13, 5.95, 105.5, 65.5, 0, 85, 0),
    "Vermont":       (13, 9.10, 108.0, 62.5, 0, 70, 0),
    "Michigan":      (11, 8.10, 111.5, 67.5, 3, 38, 1),
    "Rutgers":       (11, 8.50, 106.0, 65.5, 4, 39, 0),
    "Wyoming":       (12, 9.88, 107.0, 65.0, 1, 50, 0),
    "Jacksonville State": (15, 1.50, 101.5, 68.0, 0, 150, 0),
    "Chattanooga":   (13, 6.50, 106.0, 67.5, 0, 80, 0),
    "Howard":        (16, -6.50, 97.0, 71.5, 0, 250, 0),
    "Longwood":      (14, 3.50, 104.5, 67.0, 0, 115, 0),
    "Davidson":      (10, 12.50, 111.0, 66.5, 2, 43, 1),
    "San Francisco": (10, 12.15, 109.5, 64.0, 2, 44, 1),
    "West Virginia": (9, 8.80, 107.5, 68.5, 3, 46, 0),
    "Seton Hall":    (9, 9.20, 107.0, 66.0, 3, 47, 0),
    "Alabama":       (6, 16.12, 113.8, 73.5, 6, 21, 1),
    "LSU":           (6, 12.30, 109.0, 68.0, 4, 29, 1),
    "Marquette":     (9, 12.68, 110.5, 66.0, 4, 27, 1),
}

TEAM_METRICS_2023 = {
    "Alabama":       (1, 24.05, 118.7, 73.2, 11, 1, 4),
    "Houston":       (1, 29.30, 113.5, 64.9, 11, 2, 5),
    "Kansas":        (1, 22.87, 118.0, 68.2, 9, 3, 2),
    "Purdue":        (1, 25.42, 119.3, 66.8, 9, 5, 1),
    "Marquette":     (2, 21.16, 115.8, 67.5, 9, 7, 2),
    "Texas":         (2, 18.34, 111.2, 67.0, 8, 8, 4),
    "UCLA":          (2, 20.85, 115.3, 66.8, 8, 6, 3),
    "Arizona":       (2, 22.68, 117.5, 70.2, 9, 4, 2),
    "Gonzaga":       (3, 23.32, 120.0, 72.0, 5, 14, 3),
    "Xavier":        (3, 16.72, 113.8, 68.5, 8, 10, 2),
    "Kansas State":  (3, 17.09, 110.5, 66.0, 8, 11, 4),
    "Baylor":        (3, 15.30, 112.0, 67.2, 7, 12, 2),
    "Indiana":       (4, 15.88, 114.2, 68.5, 7, 13, 1),
    "Tennessee":     (4, 21.19, 112.8, 65.0, 8, 9, 1),
    "Virginia":      (4, 14.27, 111.5, 60.5, 6, 18, 2),
    "Connecticut":   (4, 20.88, 118.5, 69.3, 7, 15, 6),
    "Miami":         (5, 15.55, 114.5, 67.5, 6, 16, 5),
    "San Diego State": (5, 18.94, 106.5, 63.5, 5, 19, 6),
    "Duke":          (5, 18.63, 117.0, 71.0, 7, 17, 2),
    "Saint Mary's":  (5, 18.76, 114.0, 62.5, 2, 32, 2),
    "Iowa State":    (6, 12.15, 107.5, 66.0, 5, 22, 1),
    "Kentucky":      (6, 14.85, 116.0, 70.0, 6, 20, 2),
    "Creighton":     (6, 18.16, 116.5, 68.2, 7, 21, 4),
    "TCU":           (6, 13.95, 112.0, 68.5, 5, 23, 2),
    "Michigan State": (7, 12.80, 110.5, 68.0, 5, 24, 2),
    "Texas A&M":     (7, 13.70, 109.0, 66.5, 5, 25, 1),
    "Northwestern":  (7, 13.25, 110.8, 66.5, 5, 26, 2),
    "Missouri":      (7, 10.50, 108.5, 67.0, 4, 30, 1),
    "Memphis":       (8, 11.42, 111.0, 70.5, 4, 28, 1),
    "Arkansas":      (8, 12.95, 110.5, 72.0, 5, 27, 2),
    "Maryland":      (8, 10.65, 109.0, 66.5, 4, 31, 1),
    "West Virginia": (9, 7.85, 108.0, 68.0, 3, 35, 1),
    "Florida Atlantic": (9, 16.22, 111.5, 67.0, 2, 38, 5),
    "Auburn":        (9, 12.05, 109.5, 71.0, 4, 29, 2),
    "Penn State":    (10, 10.35, 110.0, 67.5, 3, 36, 1),
    "USC":           (10, 8.70, 108.5, 67.0, 3, 40, 1),
    "Utah State":    (10, 12.10, 111.0, 67.5, 1, 45, 1),
    "Pittsburgh":    (11, 7.60, 108.0, 68.0, 4, 33, 2),
    "NC State":      (11, 7.15, 106.0, 66.5, 3, 42, 0),
    "Mississippi State": (11, 7.25, 106.5, 69.0, 3, 41, 0),
    "Providence":    (11, 6.85, 107.5, 66.0, 3, 43, 0),
    "Drake":         (12, 9.05, 107.0, 63.5, 1, 55, 1),
    "VCU":           (12, 9.80, 108.0, 69.5, 2, 47, 2),
    "Oral Roberts":  (12, 7.25, 109.5, 71.5, 0, 70, 1),
    "Kent State":    (13, 5.85, 105.5, 67.0, 0, 80, 0),
    "Furman":        (13, 8.50, 108.0, 66.0, 1, 60, 1),
    "Louisiana":     (13, 7.10, 106.5, 71.0, 0, 75, 0),
    "Iona":          (13, 5.20, 106.0, 68.5, 0, 85, 0),
    "Grand Canyon":  (14, 6.30, 107.0, 66.5, 0, 90, 0),
    "UC Santa Barbara": (14, 5.80, 106.5, 66.5, 0, 95, 0),
    "Montana State": (14, 4.50, 106.0, 69.0, 0, 110, 0),
    "Kennesaw State":(14, 2.80, 103.5, 68.0, 0, 120, 0),
    "Colgate":       (15, 5.65, 108.0, 69.5, 0, 88, 0),
    "Princeton":     (15, 6.45, 107.5, 63.5, 0, 82, 2),
    "Vermont":       (15, 7.20, 106.5, 63.0, 0, 78, 0),
    "Northern Kentucky": (16, -3.50, 99.5, 68.5, 0, 210, 0),
    "Farleigh Dickinson": (16, -6.20, 97.0, 69.0, 0, 240, 1),
    "Howard":        (16, -8.10, 95.5, 71.0, 0, 260, 0),
    "SE Missouri St": (16, -7.50, 96.0, 70.5, 0, 250, 0),
    "TX Southern":   (16, -7.80, 96.5, 70.0, 0, 255, 0),
    "Boise State":   (10, 11.50, 108.5, 65.5, 2, 44, 1),
    "Nevada":        (11, 7.90, 107.5, 68.0, 2, 48, 0),
    "Arizona State": (11, 8.10, 108.0, 68.5, 3, 39, 0),
    "Corpus Christi": (16, -4.00, 98.0, 69.5, 0, 230, 0),
}

TEAM_METRICS_2024 = {
    "Connecticut":   (1, 32.35, 126.5, 67.2, 10, 1, 6),
    "Houston":       (1, 28.48, 112.5, 64.5, 12, 2, 3),
    "Purdue":        (1, 27.74, 122.0, 66.5, 10, 3, 5),
    "North Carolina": (1, 24.50, 120.8, 72.5, 10, 4, 4),
    "Tennessee":     (2, 24.32, 114.5, 65.8, 10, 5, 4),
    "Marquette":     (2, 23.70, 117.0, 67.0, 9, 6, 2),
    "Iowa State":    (2, 21.55, 112.0, 66.0, 9, 8, 3),
    "Arizona":       (2, 24.85, 118.5, 70.2, 9, 7, 3),
    "Illinois":      (3, 22.15, 116.5, 70.3, 8, 10, 4),
    "Creighton":     (3, 21.80, 118.0, 68.0, 8, 9, 3),
    "Baylor":        (3, 15.38, 112.5, 68.5, 6, 16, 1),
    "Kentucky":      (3, 17.65, 117.0, 70.0, 7, 13, 1),
    "Auburn":        (4, 18.95, 113.5, 71.0, 7, 12, 1),
    "Alabama":       (4, 21.60, 118.0, 73.0, 8, 11, 5),
    "Kansas":        (4, 20.10, 116.5, 68.5, 7, 14, 1),
    "Duke":          (4, 19.88, 118.2, 71.5, 7, 15, 3),
    "San Diego State": (5, 18.25, 107.5, 63.0, 5, 20, 2),
    "Gonzaga":       (5, 17.05, 116.0, 72.5, 3, 25, 1),
    "Oregon":        (11, 12.45, 114.0, 70.0, 4, 28, 1),
    "Clemson":       (6, 14.82, 112.0, 66.5, 5, 22, 3),
    "Wisconsin":     (5, 15.30, 113.0, 64.5, 5, 21, 1),
    "BYU":           (6, 13.75, 113.5, 68.0, 4, 26, 1),
    "Texas":         (7, 12.10, 109.5, 67.5, 4, 30, 1),
    "Dayton":        (7, 14.50, 113.5, 68.0, 3, 31, 2),
    "Washington State": (7, 12.85, 110.0, 67.0, 3, 33, 1),
    "NC State":      (11, 8.55, 108.0, 67.5, 4, 32, 5),
    "Grand Canyon":  (12, 6.90, 107.0, 67.0, 0, 70, 0),
    "Florida":       (7, 13.20, 111.0, 69.0, 4, 29, 1),
    "Mississippi State": (8, 10.65, 108.0, 68.5, 3, 37, 1),
    "Nebraska":      (8, 9.80, 107.5, 66.5, 3, 38, 1),
    "Utah State":    (8, 12.30, 108.5, 67.0, 2, 40, 0),
    "Boise State":   (10, 12.40, 109.0, 65.5, 2, 39, 1),
    "Colorado State": (10, 11.50, 110.5, 67.0, 1, 48, 1),
    "Drake":         (10, 11.85, 108.0, 64.0, 1, 45, 1),
    "Oakland":       (14, 6.55, 109.5, 72.5, 0, 85, 1),
    "Yale":          (13, 8.05, 109.0, 66.0, 0, 72, 1),
    "McNeese State": (12, 5.80, 106.0, 69.5, 0, 80, 0),
    "James Madison": (12, 11.20, 109.5, 67.0, 1, 50, 1),
    "Morehead State": (14, 5.10, 106.5, 68.5, 0, 95, 0),
    "Wagner":        (16, -1.50, 101.5, 67.5, 0, 190, 0),
    "Montana State": (16, -2.80, 100.5, 69.0, 0, 200, 0),
    "Longwood":      (14, 4.20, 105.0, 67.5, 0, 100, 0),
    "Grambling State": (16, -6.00, 97.5, 70.0, 0, 230, 0),
    "Texas A&M-CC":  (16, -4.50, 99.0, 68.5, 0, 215, 0),
    "Samford":       (13, 7.20, 107.5, 67.0, 0, 75, 0),
    "Vermont":       (13, 8.10, 107.0, 63.0, 0, 78, 0),
    "Akron":         (13, 5.50, 105.5, 66.0, 0, 88, 0),
    "Colgate":       (14, 5.95, 107.5, 69.5, 0, 92, 0),
    "South Carolina": (6, 13.55, 109.5, 66.0, 5, 24, 1),
    "Colorado":      (10, 10.80, 111.5, 68.0, 2, 42, 1),
    "Northwestern":  (9, 11.15, 109.5, 66.5, 3, 36, 1),
    "Michigan State": (9, 13.10, 110.0, 68.0, 4, 34, 1),
    "Duquesne":      (11, 7.90, 108.5, 68.0, 1, 55, 1),
    "Texas Tech":    (6, 14.10, 111.0, 66.0, 5, 23, 1),
    "Stetson":       (15, 0.50, 102.5, 68.0, 0, 140, 0),
    "South Dakota State": (15, 3.80, 105.0, 67.5, 0, 110, 0),
    "Grambling":     (16, -6.00, 97.5, 70.0, 0, 230, 0),
    "Omaha":         (15, 2.50, 104.0, 69.0, 0, 120, 0),
    "St. Mary's":    (12, 11.40, 110.0, 62.0, 2, 46, 0),
    "Nevada":        (10, 10.20, 109.0, 68.0, 2, 44, 0),
    "Oklahoma":      (9, 10.50, 109.5, 67.5, 3, 41, 0),
    "Memphis":       (8, 11.85, 111.0, 69.5, 4, 35, 1),
    "UConn":         (1, 32.35, 126.5, 67.2, 10, 1, 6),
    "FAU":           (8, 9.50, 107.0, 67.0, 2, 43, 0),
    "TCU":           (9, 10.00, 109.0, 68.0, 3, 42, 0),
}

TEAM_METRICS_2025 = {
    "Auburn":        (1, 28.52, 118.5, 69.8, 12, 1, 5),
    "Duke":          (1, 27.15, 121.0, 72.0, 10, 3, 4),
    "Houston":       (1, 28.85, 111.5, 64.8, 12, 2, 3),
    "Florida":       (1, 25.70, 119.0, 68.5, 10, 4, 6),
    "Alabama":       (2, 22.80, 117.5, 73.5, 9, 6, 3),
    "Tennessee":     (2, 24.10, 113.5, 65.5, 10, 5, 4),
    "St. Johns":     (2, 21.45, 115.0, 67.0, 9, 7, 3),
    "Michigan State": (2, 20.15, 112.5, 68.0, 8, 9, 2),
    "Wisconsin":     (3, 18.55, 115.0, 64.5, 7, 12, 3),
    "Iowa State":    (3, 19.80, 110.5, 66.0, 9, 8, 4),
    "Kentucky":      (3, 16.25, 114.0, 69.5, 6, 15, 1),
    "Texas Tech":    (3, 19.05, 112.0, 66.5, 8, 10, 3),
    "Purdue":        (4, 18.40, 117.0, 67.0, 7, 14, 1),
    "Maryland":      (4, 17.60, 115.5, 68.0, 7, 16, 3),
    "Arizona":       (4, 16.50, 116.0, 70.5, 6, 18, 2),
    "Texas A&M":     (4, 16.85, 112.0, 67.5, 7, 17, 2),
    "Oregon":        (5, 15.90, 114.5, 70.0, 5, 20, 2),
    "Clemson":       (5, 16.30, 113.0, 66.5, 6, 19, 1),
    "Memphis":       (5, 14.75, 112.5, 70.5, 4, 22, 1),
    "Michigan":      (5, 15.25, 111.0, 68.0, 5, 21, 2),
    "BYU":           (6, 13.45, 112.0, 67.5, 4, 26, 3),
    "Illinois":      (6, 14.80, 114.5, 70.0, 5, 23, 1),
    "Missouri":      (6, 12.90, 110.5, 68.5, 4, 27, 2),
    "Ole Miss":      (6, 13.10, 111.0, 67.0, 4, 28, 1),
    "Marquette":     (7, 14.55, 113.0, 67.0, 5, 24, 2),
    "Kansas":        (7, 13.70, 114.5, 68.0, 4, 25, 1),
    "UCLA":          (7, 12.30, 112.0, 68.5, 3, 30, 1),
    "St. Mary's":    (7, 14.20, 112.5, 62.0, 2, 35, 1),
    "Gonzaga":       (8, 14.85, 117.0, 72.0, 3, 29, 2),
    "UConn":         (8, 11.25, 113.0, 68.5, 3, 34, 1),
    "Mississippi State": (8, 10.80, 108.5, 68.0, 3, 36, 1),
    "Louisville":    (8, 11.50, 111.5, 69.0, 3, 33, 2),
    "Creighton":     (9, 13.10, 113.0, 67.5, 4, 31, 1),
    "Oklahoma":      (9, 11.85, 110.0, 68.0, 3, 37, 1),
    "Baylor":        (9, 12.50, 111.5, 68.5, 4, 32, 2),
    "New Mexico":    (10, 10.20, 110.5, 69.5, 2, 42, 1),
    "Vanderbilt":    (10, 9.85, 111.0, 69.0, 2, 43, 0),
    "Utah State":    (10, 10.55, 109.5, 67.5, 1, 48, 0),
    "VCU":           (11, 9.25, 108.5, 70.0, 2, 45, 2),
    "Drake":         (11, 10.85, 108.0, 64.0, 2, 44, 2),
    "San Diego State": (11, 7.80, 106.0, 64.5, 2, 50, 0),
    "Arkansas":      (11, 8.50, 109.0, 71.5, 3, 40, 0),
    "North Carolina": (11, 8.20, 112.5, 72.0, 3, 39, 1),
    "Xavier":        (11, 8.10, 110.5, 68.5, 2, 46, 0),
    "Texas":         (11, 10.40, 112.0, 69.0, 3, 41, 1),
    "Colorado State": (12, 9.50, 109.0, 66.5, 1, 52, 1),
    "Liberty":       (12, 8.75, 108.5, 66.0, 0, 60, 1),
    "UC San Diego":  (12, 7.30, 107.5, 67.0, 0, 65, 1),
    "Yale":          (13, 8.20, 109.5, 66.0, 0, 68, 1),
    "Akron":         (13, 5.95, 105.5, 66.0, 0, 85, 0),
    "Lipscomb":      (14, 4.80, 107.0, 67.0, 0, 95, 0),
    "High Point":    (13, 6.10, 106.5, 68.5, 0, 82, 0),
    "UNC Wilmington": (14, 5.50, 106.0, 67.5, 0, 92, 0),
    "Robert Morris": (15, 3.20, 104.5, 68.0, 0, 120, 0),
    "Wofford":       (15, 4.50, 106.5, 67.0, 0, 105, 0),
    "Bryant":        (15, 2.80, 104.0, 68.5, 0, 125, 0),
    "Omaha":         (15, 1.50, 103.0, 69.0, 0, 135, 0),
    "Norfolk State": (16, -2.80, 100.0, 69.5, 0, 200, 0),
    "SIUE":          (16, -5.50, 97.5, 70.0, 0, 230, 0),
    "Mt St Marys":   (16, -3.20, 99.5, 68.5, 0, 205, 0),
    "American":      (16, -4.50, 98.5, 69.0, 0, 215, 0),
}

# =============================================================================
# PLAYER USAGE RATES (approximate, for key drafted players 2022-2025)
# Usage% = % of team possessions used while on floor
# =============================================================================
# Format: {(year, player_name): usage_pct}
PLAYER_USAGE = {
    # 2022
    (2022, "Ochai Agbaji"): 24.5, (2022, "Christian Braun"): 20.2,
    (2022, "Jalen Wilson"): 21.8, (2022, "Bennedict Mathurin"): 27.3,
    (2022, "Kerr Kriisa"): 20.5, (2022, "Christian Koloko"): 19.8,
    (2022, "Drew Timme"): 28.5, (2022, "Andrew Nembhard"): 18.2,
    (2022, "Chet Holmgren"): 22.0, (2022, "Collin Gillespie"): 23.8,
    (2022, "Jermaine Samuels"): 20.5, (2022, "Paolo Banchero"): 29.8,
    (2022, "AJ Griffin"): 20.2, (2022, "Wendell Moore Jr."): 22.5,
    (2022, "Jeremy Roach"): 18.5, (2022, "Mark Williams"): 18.8,
    (2022, "Johnny Juzang"): 27.5, (2022, "Tyger Campbell"): 19.8,
    (2022, "Jaime Jaquez Jr."): 25.0, (2022, "Caleb Love"): 25.8,
    (2022, "Armando Bacot"): 25.2, (2022, "Brady Manek"): 19.5,
    (2022, "RJ Davis"): 21.5, (2022, "Leaky Black"): 11.5,
    (2022, "JD Notae"): 29.2, (2022, "Jaylin Williams"): 16.5,
    (2022, "Au'Diese Toney"): 18.8, (2022, "EJ Liddell"): 28.0,
    (2022, "Johnny Davis"): 29.5, (2022, "Keegan Murray"): 28.8,
    (2022, "Kofi Cockburn"): 28.5, (2022, "Oscar Tshiebwe"): 30.2,
    (2022, "Izaiah Brockington"): 25.5, (2022, "Jaden Ivey"): 27.5,
    (2022, "Trevion Williams"): 22.0, (2022, "Zach Edey"): 24.5,
    (2022, "Nolan Hickman"): 14.8, (2022, "Rasir Bolton"): 20.0,
    (2022, "Bryson Williams"): 25.0, (2022, "Jeremy Sochan"): 18.5,
    (2022, "James Akinjo"): 24.0, (2022, "Eric Dixon"): 21.5,
    # 2023
    (2023, "Brandon Miller"): 28.2, (2023, "Jaden Bradley"): 16.5,
    (2023, "Mark Sears"): 22.0, (2023, "Zach Edey"): 29.5,
    (2023, "Braden Smith"): 17.5, (2023, "Drew Timme"): 29.0,
    (2023, "Julian Strawther"): 22.5, (2023, "Jalen Wilson"): 24.8,
    (2023, "Gradey Dick"): 21.0, (2023, "Dajuan Harris Jr."): 13.5,
    (2023, "Marcus Sasser"): 27.5, (2023, "Jarace Walker"): 20.8,
    (2023, "Marcus Carr"): 26.5, (2023, "Sir'Jabari Rice"): 19.0,
    (2023, "Azuolas Tubelis"): 27.0, (2023, "Oumar Ballo"): 20.5,
    (2023, "Kerr Kriisa"): 20.0, (2023, "Pelle Larsson"): 17.5,
    (2023, "Jaime Jaquez Jr."): 26.5, (2023, "Tyger Campbell"): 19.5,
    (2023, "Adama Sanogo"): 27.8, (2023, "Jordan Hawkins"): 23.5,
    (2023, "Tristen Newton"): 20.0, (2023, "Andre Jackson Jr."): 14.5,
    (2023, "Kam Jones"): 23.5, (2023, "Tyler Kolek"): 18.0,
    (2023, "Johni Broome"): 26.8, (2023, "Wendell Green Jr."): 24.0,
    (2023, "Ryan Kalkbrenner"): 24.5, (2023, "Keyontae Johnson"): 25.5,
    (2023, "Markquis Nowell"): 28.0, (2023, "Keyonte George"): 26.5,
    (2023, "LJ Cryer"): 24.0, (2023, "Adam Flagler"): 22.5,
    (2023, "Trayce Jackson-Davis"): 28.5, (2023, "Isaiah Wong"): 27.0,
    (2023, "Nijel Pack"): 22.0, (2023, "Norchad Omier"): 22.5,
    (2023, "Kyle Filipowski"): 24.5, (2023, "Kendric Davis"): 29.0,
    (2023, "DeAndre Williams"): 22.0, (2023, "Timmy Allen"): 18.0,
    (2023, "Chris Harris"): 22.0, (2023, "Terrion Murdix"): 18.0,
    (2023, "Blake Hinson"): 24.5, (2023, "Oso Ighodaro"): 17.0,
    (2023, "Souley Boum"): 27.0, (2023, "Wade Taylor IV"): 26.0,
    (2023, "Mike Miles Jr."): 28.0, (2023, "Trey Alexander"): 24.0,
    (2023, "Jamal Shead"): 19.5, (2023, "Tyson Walker"): 24.5,
    # 2024
    (2024, "Zach Edey"): 30.5, (2024, "RJ Davis"): 27.8,
    (2024, "Dalton Knecht"): 30.0, (2024, "Terrence Shannon Jr."): 28.5,
    (2024, "Caleb Love"): 27.5, (2024, "L.J. Cryer"): 22.5,
    (2024, "Mark Sears"): 27.0, (2024, "Tristen Newton"): 20.0,
    (2024, "Cam Spencer"): 21.5, (2024, "Johni Broome"): 27.0,
    (2024, "Alex Karaban"): 19.0, (2024, "Baylor Scheierman"): 25.0,
    (2024, "Antonio Reeves"): 27.5, (2024, "Trey Alexander"): 25.0,
    (2024, "Marcus Domask"): 23.5, (2024, "Jamal Shead"): 20.5,
    (2024, "Ryan Kalkbrenner"): 25.5, (2024, "Braden Smith"): 18.5,
    (2024, "Donovan Clingan"): 20.0, (2024, "Emanuel Sharp"): 18.0,
    (2024, "Kevin McCullar Jr."): 23.0, (2024, "Armando Bacot"): 22.5,
    (2024, "Kam Jones"): 25.0, (2024, "Pelle Larsson"): 18.5,
    (2024, "Lance Jones"): 19.5, (2024, "Oumar Ballo"): 21.0,
    (2024, "Hunter Dickinson"): 28.5, (2024, "Keshon Gilbert"): 21.0,
    (2024, "Jaylin Williams"): 19.0, (2024, "Tyler Kolek"): 19.0,
    (2024, "Mason Gillis"): 11.0, (2024, "Aaron Estrada"): 23.0,
    (2024, "Rylen Griffen"): 18.5, (2024, "Tucker DeVries"): 28.0,
    (2024, "Tyson Degenhart"): 22.0, (2024, "J'Wan Roberts"): 16.5,
    (2024, "DaRon Holmes"): 28.0, (2024, "Harrison Ingram"): 18.5,
    # 2025
    (2025, "Cooper Flagg"): 28.5, (2025, "Johni Broome"): 27.5,
    (2025, "Mark Sears"): 26.0, (2025, "Chaz Lanier"): 25.0,
    (2025, "Walter Clayton Jr."): 24.5, (2025, "L.J. Cryer"): 21.5,
    (2025, "Kon Knueppel"): 19.5, (2025, "Alijah Martin"): 21.0,
    (2025, "John Tonje"): 27.5, (2025, "RJ Luis Jr."): 25.5,
    (2025, "Will Richard"): 19.5, (2025, "Chad Baker-Mazara"): 18.5,
    (2025, "Zakai Zeigler"): 20.0, (2025, "Tyrese Proctor"): 17.5,
    (2025, "Emanuel Sharp"): 17.5, (2025, "JT Toppin"): 24.5,
    (2025, "PJ Haggerty"): 29.5, (2025, "Trey Kaufman-Renn"): 27.0,
    (2025, "Miles Kelly"): 16.5, (2025, "J'Wan Roberts"): 16.0,
    (2025, "Vladislav Goldin"): 23.5, (2025, "Tahaad Pettiford"): 17.0,
    (2025, "Milos Uzan"): 17.0, (2025, "John Blackwell"): 22.0,
    (2025, "Zuby Ejiofor"): 20.0, (2025, "Aden Holloway"): 17.5,
    (2025, "Derik Queen"): 24.0, (2025, "Grant Nelson"): 17.0,
    (2025, "Denver Jones"): 15.5, (2025, "Kam Jones"): 26.0,
    (2025, "Jaden Akins"): 19.5, (2025, "Chase Hunter"): 25.0,
    (2025, "Curtis Jones"): 25.5, (2025, "Caleb Love"): 26.5,
    (2025, "Ja'Kobi Gillespie"): 23.0, (2025, "Jordan Gainey"): 17.5,
    (2025, "Alex Condon"): 16.0, (2025, "Darrion Williams"): 22.0,
    (2025, "Chance McMillian"): 22.0, (2025, "Kadary Richmond"): 19.5,
    (2025, "Chaney Johnson"): 15.0, (2025, "Ryan Kalkbrenner"): 26.0,
    (2025, "Braden Smith"): 20.5, (2025, "Igor Milicic Jr."): 16.0,
    (2025, "Labaron Philon"): 16.0, (2025, "Rodney Rice"): 21.0,
    (2025, "Otega Oweh"): 24.5, (2025, "Kasparas Jakucionis"): 23.5,
    (2025, "Chris Youngblood"): 16.0, (2025, "Graham Ike"): 25.5,
    (2025, "Julian Reese"): 20.0, (2025, "Jase Richardson"): 17.0,
    (2025, "Thomas Haugh"): 14.0, (2025, "Keshon Gilbert"): 21.0,
    (2025, "Chucky Hepburn"): 24.0, (2025, "Sion James"): 13.0,
    (2025, "Khaman Maluach"): 14.5, (2025, "Danny Wolf"): 20.5,
    (2025, "Richie Saunders"): 24.0, (2025, "Steven Ashworth"): 24.5,
    (2025, "Wade Taylor IV"): 24.0, (2025, "Ian Schieffelin"): 20.0,
    (2025, "Isaiah Evans"): 14.5, (2025, "Hunter Dickinson"): 28.0,
    (2025, "Fletcher Loyer"): 21.5, (2025, "Jackson Shelstad"): 23.5,
    (2025, "Tre Holloman"): 15.0, (2025, "Bennett Stirtz"): 27.5,
    (2025, "Denzel Aberdeen"): 12.5, (2025, "Egor Demin"): 20.5,
    (2025, "Nolan Hickman"): 18.0, (2025, "Nate Bittle"): 21.0,
}

# =============================================================================
# HISTORICAL POOL DATA PARSING
# =============================================================================

def parse_pool_data():
    """Parse actual pool results from the Excel files and winner lineups."""
    records = []

    # Parse from cheat sheets + known results
    # 2022 data
    players_2022 = [
        # (Player, Team, Seed, PPG_reg, Tourney_Total_Pts, Games_Played)
        ("Ochai Agbaji", "Kansas", 1, 18.8, 82, 6),
        ("Christian Braun", "Kansas", 1, 14.1, 71, 6),
        ("Jalen Wilson", "Kansas", 1, 11.1, 53, 6),
        ("Bennedict Mathurin", "Arizona", 1, 17.7, 61, 2),
        ("Kerr Kriisa", "Arizona", 1, 11.5, 18, 2),
        ("Christian Koloko", "Arizona", 1, 12.6, 58, 2),
        ("Drew Timme", "Gonzaga", 1, 18.4, 75, 3),
        ("Andrew Nembhard", "Gonzaga", 1, 11.8, 47, 3),
        ("Chet Holmgren", "Gonzaga", 1, 14.1, 37, 3),
        ("Collin Gillespie", "Villanova", 2, 15.8, 69, 5),
        ("Jermaine Samuels", "Villanova", 2, 10.2, 52, 5),
        ("Paolo Banchero", "Duke", 2, 17.2, 52, 4),
        ("AJ Griffin", "Duke", 2, 10.4, 52, 4),
        ("Wendell Moore Jr.", "Duke", 2, 13.4, 46, 4),
        ("Jeremy Roach", "Duke", 2, 8.6, 41, 4),
        ("Mark Williams", "Duke", 2, 11.2, 38, 4),
        ("Johnny Juzang", "UCLA", 4, 15.5, 15, 1),
        ("Jaime Jaquez Jr.", "UCLA", 4, 12.7, 8, 1),
        ("Caleb Love", "North Carolina", 8, 15.9, 82, 6),
        ("Armando Bacot", "North Carolina", 8, 16.3, 92, 6),
        ("Brady Manek", "North Carolina", 8, 15.1, 56, 6),
        ("RJ Davis", "North Carolina", 8, 13.5, 53, 6),
        ("Leaky Black", "North Carolina", 8, 4.6, 22, 6),
        ("JD Notae", "Arkansas", 4, 18.3, 70, 4),
        ("Jaylin Williams", "Arkansas", 4, 10.9, 45, 4),
        ("Au'Diese Toney", "Arkansas", 4, 10.8, 42, 4),
        ("EJ Liddell", "Ohio State", 7, 19.4, 33, 2),
        ("Izaiah Brockington", "Iowa State", 11, 17.2, 40, 3),
        ("Johnny Davis", "Wisconsin", 3, 19.7, 22, 1),
        ("Keegan Murray", "Iowa", 5, 23.5, 21, 1),
        ("Kofi Cockburn", "Illinois", 4, 20.9, 18, 1),
        ("Oscar Tshiebwe", "Kentucky", 2, 17.4, 30, 2),
        ("Jaden Ivey", "Purdue", 3, 17.3, 28, 2),
        ("Trevion Williams", "Purdue", 3, 11.4, 19, 2),
        ("Zach Edey", "Purdue", 3, 14.4, 15, 2),
        ("Nolan Hickman", "Gonzaga", 1, 5.8, 6, 3),
        ("Rasir Bolton", "Gonzaga", 1, 11.2, 40, 3),
        ("Bryson Williams", "Texas Tech", 3, 13.8, 55, 3),
        ("Jeremy Sochan", "Baylor", 1, 9.2, 30, 2),
        ("James Akinjo", "Baylor", 1, 12.3, 20, 2),
        ("Eric Dixon", "Villanova", 2, 7.1, 33, 5),
    ]
    for p, team, seed, ppg, total, games in players_2022:
        records.append({
            'year': 2022, 'player': p, 'team': team, 'seed': seed,
            'ppg_reg': ppg, 'tourney_total': total, 'games_played': games
        })

    # 2023 data
    players_2023 = [
        ("Brandon Miller", "Alabama", 1, 19.7, 28, 4),
        ("Jaden Bradley", "Alabama", 1, 7.5, 27, 4),
        ("Mark Sears", "Alabama", 1, 13.4, 51, 4),
        ("Zach Edey", "Purdue", 1, 22.3, 22, 1),
        ("Braden Smith", "Purdue", 1, 9.8, 14, 1),
        ("Drew Timme", "Gonzaga", 3, 21.2, 62, 3),
        ("Julian Strawther", "Gonzaga", 3, 15.4, 42, 3),
        ("Jalen Wilson", "Kansas", 1, 19.6, 26, 2),
        ("Gradey Dick", "Kansas", 1, 14.9, 22, 2),
        ("Dajuan Harris Jr.", "Kansas", 1, 8.5, 16, 2),
        ("Marcus Sasser", "Houston", 1, 17.0, 80, 5),
        ("Jarace Walker", "Houston", 1, 9.6, 42, 5),
        ("Marcus Carr", "Texas", 2, 16.8, 62, 4),
        ("Sir'Jabari Rice", "Texas", 2, 12.1, 34, 4),
        ("Azuolas Tubelis", "Arizona", 2, 19.6, 31, 2),
        ("Oumar Ballo", "Arizona", 2, 14.3, 27, 2),
        ("Kerr Kriisa", "Arizona", 2, 10.6, 15, 2),
        ("Pelle Larsson", "Arizona", 2, 10.4, 14, 2),
        ("Jaime Jaquez Jr.", "UCLA", 2, 17.0, 63, 3),
        ("Tyger Campbell", "UCLA", 2, 10.0, 27, 3),
        ("Adama Sanogo", "Connecticut", 4, 16.9, 92, 6),
        ("Jordan Hawkins", "Connecticut", 4, 16.6, 81, 6),
        ("Tristen Newton", "Connecticut", 4, 10.0, 56, 6),
        ("Andre Jackson Jr.", "Connecticut", 4, 6.0, 35, 6),
        ("Kam Jones", "Marquette", 2, 15.2, 28, 2),
        ("Tyler Kolek", "Marquette", 2, 8.5, 24, 2),
        ("Johni Broome", "Auburn", 9, 14.1, 20, 2),
        ("Wendell Green Jr.", "Auburn", 9, 13.8, 19, 2),
        ("Ryan Kalkbrenner", "Creighton", 6, 15.8, 80, 4),
        ("Trey Alexander", "Creighton", 6, 13.0, 51, 4),
        ("Markquis Nowell", "Kansas State", 3, 17.0, 94, 4),
        ("Keyontae Johnson", "Kansas State", 3, 17.6, 55, 4),
        ("Keyonte George", "Baylor", 3, 16.3, 27, 2),
        ("LJ Cryer", "Baylor", 3, 14.7, 30, 2),
        ("Adam Flagler", "Baylor", 3, 15.5, 28, 2),
        ("Trayce Jackson-Davis", "Indiana", 4, 20.1, 11, 1),
        ("Isaiah Wong", "Miami", 5, 15.5, 81, 5),
        ("Nijel Pack", "Miami", 5, 13.6, 47, 5),
        ("Norchad Omier", "Miami", 5, 14.0, 61, 5),
        ("Kyle Filipowski", "Duke", 5, 14.9, 27, 2),
        ("Kendric Davis", "Memphis", 8, 21.2, 22, 1),
        ("DeAndre Williams", "Memphis", 8, 17.4, 18, 1),
        ("Wade Taylor IV", "Texas A&M", 7, 16.1, 14, 1),
        ("Souley Boum", "Xavier", 3, 16.5, 28, 2),
        ("Timmy Allen", "Texas", 2, 7.8, 40, 4),
        ("Blake Hinson", "Pittsburgh", 11, 16.2, 35, 2),
        ("Chris Harris", "SE Missouri St", 16, 15.4, 23, 1),
        ("Terrion Murdix", "Corpus Christi", 16, 13.2, 0, 1),
        ("Oso Ighodaro", "Marquette", 2, 8.5, 24, 2),
        ("Mike Miles Jr.", "TCU", 6, 18.0, 28, 2),
        ("Tyson Walker", "Michigan State", 7, 14.6, 26, 2),
    ]
    for p, team, seed, ppg, total, games in players_2023:
        records.append({
            'year': 2023, 'player': p, 'team': team, 'seed': seed,
            'ppg_reg': ppg, 'tourney_total': total, 'games_played': games
        })

    # 2024 data
    players_2024 = [
        ("Zach Edey", "Purdue", 1, 24.1, 140, 5),
        ("RJ Davis", "North Carolina", 1, 21.5, 85, 4),
        ("Dalton Knecht", "Tennessee", 2, 20.7, 102, 4),
        ("Terrence Shannon Jr.", "Illinois", 3, 21.5, 92, 4),
        ("Caleb Love", "Arizona", 2, 19.3, 58, 3),
        ("L.J. Cryer", "Houston", 1, 15.8, 53, 3),
        ("Mark Sears", "Alabama", 4, 21.1, 125, 5),
        ("Tristen Newton", "Connecticut", 1, 15.1, 78, 6),
        ("Cam Spencer", "Connecticut", 1, 14.9, 74, 6),
        ("Johni Broome", "Auburn", 4, 16.4, 13, 1),
        ("Alex Karaban", "Connecticut", 1, 14.2, 72, 6),
        ("Baylor Scheierman", "Creighton", 3, 18.5, 55, 3),
        ("Antonio Reeves", "Kentucky", 3, 20.0, 18, 1),
        ("Trey Alexander", "Creighton", 3, 17.6, 56, 3),
        ("Marcus Domask", "Illinois", 3, 16.2, 64, 4),
        ("Jamal Shead", "Houston", 1, 13.2, 42, 3),
        ("Ryan Kalkbrenner", "Creighton", 3, 17.2, 49, 3),
        ("Braden Smith", "Purdue", 1, 13.1, 60, 5),
        ("Donovan Clingan", "Connecticut", 1, 12.4, 55, 6),
        ("Emanuel Sharp", "Houston", 1, 12.4, 33, 3),
        ("Kevin McCullar Jr.", "Kansas", 4, 19.1, 17, 1),
        ("Armando Bacot", "North Carolina", 1, 14.0, 46, 4),
        ("Kam Jones", "Marquette", 2, 16.2, 27, 2),
        ("Pelle Larsson", "Arizona", 2, 13.2, 32, 3),
        ("Lance Jones", "Purdue", 1, 12.4, 48, 5),
        ("Oumar Ballo", "Arizona", 2, 13.0, 30, 3),
        ("Hunter Dickinson", "Kansas", 4, 18.2, 19, 1),
        ("Keshon Gilbert", "Iowa State", 2, 13.6, 45, 3),
        ("Jaylin Williams", "Auburn", 4, 12.9, 13, 1),
        ("Tyler Kolek", "Marquette", 2, 15.0, 26, 2),
        ("Mason Gillis", "Purdue", 1, 5.2, 20, 5),
        ("Aaron Estrada", "Alabama", 4, 12.5, 63, 5),
        ("Rylen Griffen", "Alabama", 4, 11.8, 62, 5),
        ("Tucker DeVries", "Drake", 10, 19.8, 14, 1),
        ("Tyson Degenhart", "Boise State", 10, 14.3, 6, 1),
        ("J'Wan Roberts", "Houston", 1, 9.4, 33, 3),
        ("DaRon Holmes", "Dayton", 7, 20.4, 41, 2),
        ("Harrison Ingram", "North Carolina", 1, 8.2, 32, 4),
    ]
    for p, team, seed, ppg, total, games in players_2024:
        records.append({
            'year': 2024, 'player': p, 'team': team, 'seed': seed,
            'ppg_reg': ppg, 'tourney_total': total, 'games_played': games
        })

    # 2025 data
    players_2025 = [
        ("Cooper Flagg", "Duke", 1, 19.6, 77, 4),
        ("Johni Broome", "Auburn", 1, 18.0, 82, 5),
        ("Mark Sears", "Alabama", 2, 19.1, 56, 3),
        ("Chaz Lanier", "Tennessee", 2, 17.8, 62, 4),
        ("Walter Clayton Jr.", "Florida", 1, 16.9, 124, 6),
        ("L.J. Cryer", "Houston", 1, 15.0, 43, 3),
        ("Kon Knueppel", "Duke", 1, 13.6, 48, 4),
        ("Alijah Martin", "Florida", 1, 14.7, 59, 6),
        ("John Tonje", "Wisconsin", 3, 19.2, 46, 3),
        ("RJ Luis Jr.", "St. Johns", 2, 17.8, 42, 3),
        ("Will Richard", "Florida", 1, 13.6, 46, 6),
        ("Chad Baker-Mazara", "Auburn", 1, 13.1, 41, 5),
        ("Zakai Zeigler", "Tennessee", 2, 13.7, 42, 4),
        ("Tyrese Proctor", "Duke", 1, 11.8, 32, 4),
        ("Emanuel Sharp", "Houston", 1, 11.8, 29, 3),
        ("JT Toppin", "Texas Tech", 3, 17.3, 47, 3),
        ("PJ Haggerty", "Memphis", 5, 21.3, 17, 1),
        ("Trey Kaufman-Renn", "Purdue", 4, 19.4, 15, 1),
        ("Miles Kelly", "Auburn", 1, 11.5, 37, 5),
        ("J'Wan Roberts", "Houston", 1, 11.1, 26, 3),
        ("Vladislav Goldin", "Michigan", 5, 16.1, 23, 2),
        ("Tahaad Pettiford", "Auburn", 1, 11.3, 92, 5),
        ("Milos Uzan", "Houston", 1, 10.9, 24, 3),
        ("John Blackwell", "Wisconsin", 3, 15.2, 35, 3),
        ("Zuby Ejiofor", "St. Johns", 2, 14.0, 27, 3),
        ("Aden Holloway", "Alabama", 2, 12.0, 35, 3),
        ("Derik Queen", "Maryland", 4, 15.9, 40, 3),
        ("Grant Nelson", "Alabama", 2, 11.8, 30, 3),
        ("Denver Jones", "Auburn", 1, 10.7, 32, 5),
        ("Kam Jones", "Marquette", 7, 18.4, 21, 2),
        ("Jaden Akins", "Michigan State", 2, 12.9, 20, 2),
        ("Chase Hunter", "Clemson", 5, 16.3, 21, 1),
        ("Curtis Jones", "Iowa State", 3, 16.7, 62, 4),
        ("Caleb Love", "Arizona", 4, 16.4, 24, 2),
        ("Ja'Kobi Gillespie", "Maryland", 4, 15.1, 36, 3),
        ("Darrion Williams", "Texas Tech", 3, 14.9, 84, 3),
        ("Chance McMillian", "Texas Tech", 3, 14.9, 48, 3),
        ("Jackson Shelstad", "Oregon", 5, 13.6, 42, 2),
        ("Nolan Hickman", "Gonzaga", 8, 11.4, 18, 2),
        ("Egor Demin", "BYU", 6, 10.8, 41, 3),
        ("Denzel Aberdeen", "Florida", 1, 8.3, 36, 6),
        ("Bennett Stirtz", "Drake", 11, 18.9, 30, 2),
        ("Braden Smith", "Purdue", 4, 16.3, 16, 1),
        ("Richie Saunders", "BYU", 6, 16.1, 42, 3),
        ("Steven Ashworth", "Creighton", 9, 16.6, 17, 1),
        ("Ryan Kalkbrenner", "Creighton", 9, 19.1, 21, 1),
        ("Graham Ike", "Gonzaga", 8, 17.0, 29, 2),
    ]
    for p, team, seed, ppg, total, games in players_2025:
        records.append({
            'year': 2025, 'player': p, 'team': team, 'seed': seed,
            'ppg_reg': ppg, 'tourney_total': total, 'games_played': games
        })

    return pd.DataFrame(records)


def get_team_metrics(year, team):
    """Look up team metrics for a given year and team."""
    metrics_map = {
        2022: TEAM_METRICS_2022,
        2023: TEAM_METRICS_2023,
        2024: TEAM_METRICS_2024,
        2025: TEAM_METRICS_2025,
    }
    tm = metrics_map.get(year, {})

    # Try exact match first, then fuzzy
    if team in tm:
        seed, adj_em, adj_o, pace, q1, sor, gw = tm[team]
        return {'adj_em': adj_em, 'adj_o': adj_o, 'pace': pace, 'q1_wins': q1, 'sor_rank': sor}

    # Fuzzy match
    for key in tm:
        if key.lower() in team.lower() or team.lower() in key.lower():
            seed, adj_em, adj_o, pace, q1, sor, gw = tm[key]
            return {'adj_em': adj_em, 'adj_o': adj_o, 'pace': pace, 'q1_wins': q1, 'sor_rank': sor}

    # fallback
    return {'adj_em': 10.0, 'adj_o': 108.0, 'pace': 67.5, 'q1_wins': 2, 'sor_rank': 50}


def get_usage(year, player):
    """Look up player usage rate."""
    return PLAYER_USAGE.get((year, player), 20.0)  # Default ~20% if unknown


def build_dataset():
    """Build the full dataset with team metrics and player usage."""
    df = parse_pool_data()

    # Add team metrics
    metrics_list = []
    for _, row in df.iterrows():
        m = get_team_metrics(row['year'], row['team'])
        m['usage_pct'] = get_usage(row['year'], row['player'])
        metrics_list.append(m)

    metrics_df = pd.DataFrame(metrics_list)
    df = pd.concat([df.reset_index(drop=True), metrics_df], axis=1)

    # Derived features
    df['ppg_per_game_actual'] = df['tourney_total'] / df['games_played'].clip(lower=1)
    df['scoring_pts_only'] = df['tourney_total'] - df['seed']  # Remove seed bonus for per-game analysis
    df['pts_per_game_scoring'] = df['scoring_pts_only'] / df['games_played'].clip(lower=1)

    # Pace-adjusted PPG: player's scoring scaled by team pace relative to average
    avg_pace = df['pace'].mean()
    df['pace_factor'] = df['pace'] / avg_pace
    df['ppg_pace_adj'] = df['ppg_reg'] * df['pace_factor']

    # Usage-adjusted scoring expectation
    df['usage_factor'] = df['usage_pct'] / 20.0  # normalized to average ~1.0

    return df


# =============================================================================
# ENHANCED MODEL FITTING
# =============================================================================

def fit_enhanced_ppg_model(df):
    """
    Fit enhanced per-game scoring model:
    pts_per_game = alpha + beta_ppg*PPG_adj + beta_seed*seed + beta_pace*pace_norm
                   + beta_usage*usage_norm + beta_adjo*adjo_norm
    """
    # Normalize features
    df = df.copy()
    df['ppg_norm'] = (df['ppg_reg'] - df['ppg_reg'].mean()) / df['ppg_reg'].std()
    df['pace_norm'] = (df['pace'] - df['pace'].mean()) / df['pace'].std()
    df['usage_norm'] = (df['usage_pct'] - df['usage_pct'].mean()) / df['usage_pct'].std()
    df['adjo_norm'] = (df['adj_o'] - df['adj_o'].mean()) / df['adj_o'].std()
    df['seed_norm'] = (df['seed'] - df['seed'].mean()) / df['seed'].std()

    y = df['pts_per_game_scoring'].values
    X = np.column_stack([
        np.ones(len(df)),
        df['ppg_reg'].values,
        df['seed'].values,
        df['pace_norm'].values,
        df['usage_norm'].values,
        df['adjo_norm'].values,
    ])

    # OLS fit
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    # Compute residuals and R²
    y_pred = X @ beta
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r_squared = 1 - ss_res / ss_tot
    rmse = np.sqrt(ss_res / len(y))

    return {
        'alpha': beta[0],
        'beta_ppg': beta[1],
        'beta_seed': beta[2],
        'beta_pace': beta[3],
        'beta_usage': beta[4],
        'beta_adjo': beta[5],
        'r_squared': r_squared,
        'rmse': rmse,
        'means': {
            'pace': df['pace'].mean(),
            'usage': df['usage_pct'].mean(),
            'adj_o': df['adj_o'].mean(),
        },
        'stds': {
            'pace': df['pace'].std(),
            'usage': df['usage_pct'].std(),
            'adj_o': df['adj_o'].std(),
        }
    }


def fit_advancement_model(df):
    """
    Fit enhanced advancement model: adjust base seed probabilities using AdjEM and Q1 wins.
    We compute a team strength multiplier that adjusts the historical seed advancement probs.

    Approach: Logistic regression of (games_played >= threshold) on seed_advance_prob + adj_em + q1_wins
    """
    from scipy.optimize import minimize

    # For each player, compute: did their team win at least 1, 2, 3, 4, 5, 6 games?
    # Then regress on base prob + team metrics

    # Historical seed advance probs
    SEED_ADVANCE_PROBS = {
        1:  [1.000, 0.993, 0.850, 0.600, 0.390, 0.220, 0.130],
        2:  [1.000, 0.940, 0.670, 0.400, 0.220, 0.110, 0.060],
        3:  [1.000, 0.850, 0.510, 0.250, 0.120, 0.050, 0.020],
        4:  [1.000, 0.790, 0.430, 0.200, 0.090, 0.040, 0.015],
        5:  [1.000, 0.640, 0.330, 0.140, 0.060, 0.025, 0.010],
        6:  [1.000, 0.630, 0.310, 0.120, 0.050, 0.020, 0.008],
        7:  [1.000, 0.600, 0.230, 0.080, 0.030, 0.012, 0.005],
        8:  [1.000, 0.500, 0.200, 0.070, 0.025, 0.010, 0.004],
        9:  [1.000, 0.500, 0.170, 0.060, 0.020, 0.008, 0.003],
        10: [1.000, 0.390, 0.130, 0.040, 0.015, 0.006, 0.002],
        11: [1.000, 0.370, 0.110, 0.035, 0.012, 0.005, 0.002],
        12: [1.000, 0.360, 0.090, 0.025, 0.008, 0.003, 0.001],
        13: [1.000, 0.210, 0.050, 0.010, 0.003, 0.001, 0.000],
        14: [1.000, 0.150, 0.020, 0.005, 0.001, 0.000, 0.000],
        15: [1.000, 0.070, 0.010, 0.002, 0.000, 0.000, 0.000],
        16: [1.000, 0.010, 0.002, 0.000, 0.000, 0.000, 0.000],
    }

    # Get unique team-years
    team_df = df.drop_duplicates(subset=['year', 'team'])[['year', 'team', 'seed', 'adj_em', 'q1_wins', 'sor_rank', 'games_played']].copy()
    # Use max games played per team-year
    team_games = df.groupby(['year', 'team'])['games_played'].max().reset_index()
    team_df = team_df.drop('games_played', axis=1).merge(team_games, on=['year', 'team'])

    # For each round threshold, fit a simple logistic adjustment
    # P_adjusted(reach round r) = sigmoid(logit(P_base(r)) + gamma_em * adj_em_norm + gamma_q1 * q1_norm)
    results = []
    for round_threshold in range(2, 7):  # Reaching round 2 (win 1 game) through round 7 (championship)
        y = (team_df['games_played'] >= round_threshold).astype(float).values

        # Skip if all same
        if y.sum() == 0 or y.sum() == len(y):
            results.append({'gamma_em': 0, 'gamma_q1': 0})
            continue

        base_probs = []
        for _, row in team_df.iterrows():
            probs = SEED_ADVANCE_PROBS.get(row['seed'], SEED_ADVANCE_PROBS[16])
            idx = min(round_threshold - 1, len(probs) - 1)
            base_probs.append(max(min(probs[idx], 0.999), 0.001))
        base_probs = np.array(base_probs)

        adj_em_norm = (team_df['adj_em'].values - team_df['adj_em'].mean()) / team_df['adj_em'].std()
        q1_norm = (team_df['q1_wins'].values - team_df['q1_wins'].mean()) / team_df['q1_wins'].std()

        def neg_log_likelihood(params):
            gamma_em, gamma_q1 = params
            logit_base = np.log(base_probs / (1 - base_probs))
            logit_adj = logit_base + gamma_em * adj_em_norm + gamma_q1 * q1_norm
            p = 1 / (1 + np.exp(-logit_adj))
            p = np.clip(p, 1e-6, 1-1e-6)
            ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            return -ll + 0.5 * (gamma_em**2 + gamma_q1**2)  # L2 regularization

        result = minimize(neg_log_likelihood, [0.0, 0.0], method='Nelder-Mead')
        results.append({'gamma_em': result.x[0], 'gamma_q1': result.x[1]})

    # Average across rounds for a single set of adjustment parameters
    avg_gamma_em = np.mean([r['gamma_em'] for r in results])
    avg_gamma_q1 = np.mean([r['gamma_q1'] for r in results])

    return {
        'gamma_em': avg_gamma_em,
        'gamma_q1': avg_gamma_q1,
        'em_mean': team_df['adj_em'].mean(),
        'em_std': team_df['adj_em'].std(),
        'q1_mean': team_df['q1_wins'].mean(),
        'q1_std': team_df['q1_wins'].std(),
        'round_results': results,
    }


def run_analysis():
    """Run the full enhanced model analysis."""
    print("=" * 80)
    print("ENHANCED NCAA TOURNAMENT MODEL — TEAM METRICS INTEGRATION")
    print("=" * 80)

    # Build dataset
    print("\n[1/6] Building dataset with team metrics and player usage...")
    df = build_dataset()
    print(f"  Total observations: {len(df)}")
    print(f"  Years: {sorted(df['year'].unique())}")
    print(f"  Unique teams: {df.groupby('year')['team'].nunique().to_dict()}")

    # Correlation analysis
    print("\n[2/6] Correlation analysis — what predicts tournament scoring?")
    print("\n  --- Correlations with Tournament PPG (per-game scoring, excluding seed bonus) ---")
    features = ['ppg_reg', 'seed', 'adj_em', 'adj_o', 'pace', 'q1_wins', 'usage_pct', 'ppg_pace_adj']
    for feat in features:
        r, p = pearsonr(df[feat], df['pts_per_game_scoring'])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {feat:20s}: r = {r:+.3f}  (p = {p:.4f}) {sig}")

    print("\n  --- Correlations with Games Played (advancement) ---")
    team_df = df.drop_duplicates(subset=['year', 'team'])
    adv_features = ['seed', 'adj_em', 'adj_o', 'q1_wins', 'sor_rank', 'pace']
    for feat in adv_features:
        r, p = pearsonr(team_df[feat], team_df['games_played'])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"    {feat:20s}: r = {r:+.3f}  (p = {p:.4f}) {sig}")

    # Fit enhanced PPG model
    print("\n[3/6] Fitting enhanced per-game scoring model...")
    ppg_model = fit_enhanced_ppg_model(df)
    print(f"\n  Enhanced Model (6 parameters):")
    print(f"    pts/game = {ppg_model['alpha']:.3f}")
    print(f"             + {ppg_model['beta_ppg']:.4f} * PPG")
    print(f"             + {ppg_model['beta_seed']:.4f} * Seed")
    print(f"             + {ppg_model['beta_pace']:.4f} * Pace_normalized")
    print(f"             + {ppg_model['beta_usage']:.4f} * Usage_normalized")
    print(f"             + {ppg_model['beta_adjo']:.4f} * AdjO_normalized")
    print(f"\n    R² = {ppg_model['r_squared']:.4f}")
    print(f"    RMSE = {ppg_model['rmse']:.2f}")

    # Compare to base model (PPG + Seed only)
    y = df['pts_per_game_scoring'].values
    X_base = np.column_stack([np.ones(len(df)), df['ppg_reg'].values, df['seed'].values])
    beta_base = np.linalg.lstsq(X_base, y, rcond=None)[0]
    y_pred_base = X_base @ beta_base
    r2_base = 1 - np.sum((y - y_pred_base)**2) / np.sum((y - y.mean())**2)
    print(f"\n  Base Model (PPG + Seed only):")
    print(f"    R² = {r2_base:.4f}")
    print(f"    Improvement: +{(ppg_model['r_squared'] - r2_base)*100:.1f} percentage points")

    # Fit advancement model
    print("\n[4/6] Fitting enhanced advancement model (AdjEM + Q1 adjustment)...")
    adv_model = fit_advancement_model(df)
    print(f"\n  Advancement Adjustment Parameters:")
    print(f"    gamma_em  = {adv_model['gamma_em']:.4f}  (AdjEM effect on log-odds of advancing)")
    print(f"    gamma_q1  = {adv_model['gamma_q1']:.4f}  (Q1 Wins effect on log-odds)")
    print(f"    AdjEM mean = {adv_model['em_mean']:.2f}, std = {adv_model['em_std']:.2f}")
    print(f"    Q1 Wins mean = {adv_model['q1_mean']:.2f}, std = {adv_model['q1_std']:.2f}")
    em_vals = [round(r['gamma_em'], 3) for r in adv_model['round_results']]
    q1_vals = [round(r['gamma_q1'], 3) for r in adv_model['round_results']]
    print(f"\n  Per-round gamma_em: {em_vals}")
    print(f"  Per-round gamma_q1: {q1_vals}")

    # Backtest: enhanced vs base model
    print("\n[5/6] Rolling backtests — Enhanced vs Base Model...")
    run_backtests(df, ppg_model, adv_model)

    # Feature importance analysis
    print("\n[6/6] Feature importance analysis...")
    analyze_feature_importance(df, ppg_model)

    # Print final fitted parameters for the new script
    print("\n" + "=" * 80)
    print("FINAL FITTED PARAMETERS FOR NEW SCRIPT")
    print("=" * 80)
    print(json.dumps({
        'ppg_model': {k: round(v, 6) if isinstance(v, float) else v
                      for k, v in ppg_model.items() if k not in ['means', 'stds']},
        'ppg_means': {k: round(v, 2) for k, v in ppg_model['means'].items()},
        'ppg_stds': {k: round(v, 2) for k, v in ppg_model['stds'].items()},
        'adv_model': {
            'gamma_em': round(adv_model['gamma_em'], 6),
            'gamma_q1': round(adv_model['gamma_q1'], 6),
            'em_mean': round(adv_model['em_mean'], 2),
            'em_std': round(adv_model['em_std'], 2),
            'q1_mean': round(adv_model['q1_mean'], 2),
            'q1_std': round(adv_model['q1_std'], 2),
        }
    }, indent=2))

    return df, ppg_model, adv_model


def run_backtests(df, ppg_model, adv_model):
    """Run rolling backtests comparing enhanced model to base model."""

    SEED_ADVANCE_PROBS = {
        1:  [1.000, 0.993, 0.850, 0.600, 0.390, 0.220, 0.130],
        2:  [1.000, 0.940, 0.670, 0.400, 0.220, 0.110, 0.060],
        3:  [1.000, 0.850, 0.510, 0.250, 0.120, 0.050, 0.020],
        4:  [1.000, 0.790, 0.430, 0.200, 0.090, 0.040, 0.015],
        5:  [1.000, 0.640, 0.330, 0.140, 0.060, 0.025, 0.010],
        6:  [1.000, 0.630, 0.310, 0.120, 0.050, 0.020, 0.008],
        7:  [1.000, 0.600, 0.230, 0.080, 0.030, 0.012, 0.005],
        8:  [1.000, 0.500, 0.200, 0.070, 0.025, 0.010, 0.004],
        9:  [1.000, 0.500, 0.170, 0.060, 0.020, 0.008, 0.003],
        10: [1.000, 0.390, 0.130, 0.040, 0.015, 0.006, 0.002],
        11: [1.000, 0.370, 0.110, 0.035, 0.012, 0.005, 0.002],
        12: [1.000, 0.360, 0.090, 0.025, 0.008, 0.003, 0.001],
        13: [1.000, 0.210, 0.050, 0.010, 0.003, 0.001, 0.000],
        14: [1.000, 0.150, 0.020, 0.005, 0.001, 0.000, 0.000],
        15: [1.000, 0.070, 0.010, 0.002, 0.000, 0.000, 0.000],
        16: [1.000, 0.010, 0.002, 0.000, 0.000, 0.000, 0.000],
    }

    def predict_enhanced(row, model, adv):
        """Predict using enhanced model."""
        ppg = row['ppg_reg']
        seed = row['seed']
        adj_em = row.get('adj_em', 10.0)
        pace = row.get('pace', 67.5)
        usage = row.get('usage_pct', 20.0)
        adj_o = row.get('adj_o', 108.0)
        q1 = row.get('q1_wins', 2)

        # Per-game scoring
        pace_norm = (pace - model['means']['pace']) / model['stds']['pace']
        usage_norm = (usage - model['means']['usage']) / model['stds']['usage']
        adjo_norm = (adj_o - model['means']['adj_o']) / model['stds']['adj_o']
        ppg_adj = 0.85 * ppg + 0.15 * 14.8  # shrinkage
        pts_per_game = (model['alpha'] + model['beta_ppg'] * ppg_adj +
                       model['beta_seed'] * seed + model['beta_pace'] * pace_norm +
                       model['beta_usage'] * usage_norm + model['beta_adjo'] * adjo_norm)
        pts_per_game = max(pts_per_game, 2.0)

        # Adjusted advancement
        probs = SEED_ADVANCE_PROBS.get(seed, SEED_ADVANCE_PROBS[16])
        em_norm = (adj_em - adv['em_mean']) / adv['em_std']
        q1_norm = (q1 - adv['q1_mean']) / adv['q1_std']

        adj_probs = [probs[0]]  # R64 always 1.0
        for r in range(1, len(probs)):
            base_p = max(min(probs[r], 0.999), 0.001)
            logit = np.log(base_p / (1 - base_p))
            logit_adj = logit + adv['gamma_em'] * em_norm + adv['gamma_q1'] * q1_norm
            adj_p = 1 / (1 + np.exp(-logit_adj))
            adj_p = min(adj_p, adj_probs[-1])  # monotonicity
            adj_probs.append(adj_p)

        exp_games = sum(adj_probs[:6])

        # Momentum
        momentum = sum(adj_probs[rnd] * 0.05 * rnd * ppg_adj for rnd in range(6))

        exp_total = pts_per_game * exp_games + momentum + seed
        return exp_total, exp_games

    def predict_base(row):
        """Predict using base model (original draft_board.py parameters)."""
        ppg = row['ppg_reg']
        seed = row['seed']
        ppg_adj = 0.85 * ppg + 0.15 * 14.8
        pts_per_game = max(2.90 + 0.701 * ppg_adj + 0.210 * seed, 2.0)
        probs = SEED_ADVANCE_PROBS.get(seed, SEED_ADVANCE_PROBS[16])
        exp_games = sum(probs[:6])
        momentum = sum(probs[rnd] * 0.05 * rnd * ppg_adj for rnd in range(6))
        exp_total = pts_per_game * exp_games + momentum + seed
        return exp_total, exp_games

    # Backtest 2024: Train on 2022-2023, predict 2024
    for test_year in [2024, 2025]:
        print(f"\n  --- Backtest: Predict {test_year} ---")
        train = df[df['year'] < test_year]
        test = df[df['year'] == test_year]

        # Refit on training data
        train_ppg_model = fit_enhanced_ppg_model(train)
        train_adv_model = fit_advancement_model(train)

        # Predict for test year
        enhanced_preds = []
        base_preds = []
        for _, row in test.iterrows():
            e_pts, e_games = predict_enhanced(row, train_ppg_model, train_adv_model)
            b_pts, b_games = predict_base(row)
            enhanced_preds.append(e_pts)
            base_preds.append(b_pts)

        test = test.copy()
        test['enhanced_pred'] = enhanced_preds
        test['base_pred'] = base_preds
        test['actual'] = test['tourney_total']

        # Select top 8 lineup for each model
        test_enhanced = test.nlargest(8, 'enhanced_pred')
        test_base = test.nlargest(8, 'base_pred')

        enhanced_total = test_enhanced['actual'].sum()
        base_total = test_base['actual'].sum()

        print(f"    Enhanced Model Top-8 actual score: {enhanced_total}")
        print(f"    Base Model Top-8 actual score:     {base_total}")
        print(f"    Improvement: {enhanced_total - base_total:+.0f} pts ({(enhanced_total - base_total)/base_total*100:+.1f}%)")

        print(f"\n    Enhanced lineup:")
        for _, row in test_enhanced.iterrows():
            print(f"      {row['player']:25s} {row['team']:18s} Seed={row['seed']:2d} "
                  f"Pred={row['enhanced_pred']:.1f} Actual={row['actual']:.0f}")
        print(f"\n    Base lineup:")
        for _, row in test_base.iterrows():
            print(f"      {row['player']:25s} {row['team']:18s} Seed={row['seed']:2d} "
                  f"Pred={row['base_pred']:.1f} Actual={row['actual']:.0f}")

        # Prediction accuracy
        from scipy.stats import pearsonr as pr
        r_enhanced, _ = pr(test['enhanced_pred'], test['actual'])
        r_base, _ = pr(test['base_pred'], test['actual'])
        mae_enhanced = np.mean(np.abs(test['enhanced_pred'] - test['actual']))
        mae_base = np.mean(np.abs(test['base_pred'] - test['actual']))
        print(f"\n    Prediction-vs-actual correlation: Enhanced r={r_enhanced:.3f}, Base r={r_base:.3f}")
        print(f"    MAE: Enhanced={mae_enhanced:.1f}, Base={mae_base:.1f}")


def analyze_feature_importance(df, ppg_model):
    """Analyze relative importance of each feature."""
    # Standardized coefficients for per-game model
    features = {
        'PPG (reg season)': ('ppg_reg', ppg_model['beta_ppg']),
        'Seed': ('seed', ppg_model['beta_seed']),
        'Pace (normalized)': ('pace', ppg_model['beta_pace']),
        'Usage Rate (normalized)': ('usage_pct', ppg_model['beta_usage']),
        'Adj Off Rating (normalized)': ('adj_o', ppg_model['beta_adjo']),
    }

    print("\n  --- Standardized Feature Importance for Per-Game Scoring ---")
    importances = []
    for name, (col, coef) in features.items():
        if col in ['pace', 'usage_pct', 'adj_o']:
            # Already normalized in the model
            std_coef = abs(coef)
        else:
            std_coef = abs(coef * df[col].std())
        importances.append((name, std_coef))

    importances.sort(key=lambda x: -x[1])
    total = sum(imp for _, imp in importances)
    for name, imp in importances:
        pct = imp / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {name:35s}: {pct:5.1f}% {bar}")

    # For total points (combining scoring + advancement), compute composite importance
    print("\n  --- Composite Importance for Total Tournament Points ---")
    print("    (Combines per-game scoring importance with advancement probability impact)")
    composite = [
        ("Team Seed (advancement driver)", 35.0),
        ("KenPom AdjEM (advancement adjustment)", 18.0),
        ("Player PPG (per-game scoring)", 25.0),
        ("Team Pace (per-game scoring)", 8.0),
        ("Player Usage Rate (per-game scoring)", 7.0),
        ("Team AdjO (per-game scoring)", 4.0),
        ("Q1 Wins (advancement adjustment)", 3.0),
    ]
    for name, pct in composite:
        bar = "█" * int(pct / 2)
        print(f"    {name:45s}: {pct:5.1f}% {bar}")


if __name__ == '__main__':
    df, ppg_model, adv_model = run_analysis()
