-- Fix prediction_correct field based on actual probabilities
-- This recalculates all prediction_correct values for completed games

-- NFL predictions
UPDATE nfl_model_predictions
SET prediction_correct = (
  CASE 
    -- If home team was predicted to win (home_ml_prob > 0.5) and home team actually won
    WHEN CAST(home_ml_prob AS FLOAT) > 0.5 AND actual_winner = 'home' THEN true
    -- If away team was predicted to win (away_ml_prob > 0.5) and away team actually won
    WHEN CAST(away_ml_prob AS FLOAT) > 0.5 AND actual_winner = 'away' THEN true
    -- If home team was predicted to win (home_ml_prob > 0.5) but away team won
    WHEN CAST(home_ml_prob AS FLOAT) > 0.5 AND actual_winner = 'away' THEN false
    -- If away team was predicted to win (away_ml_prob > 0.5) but home team won
    WHEN CAST(away_ml_prob AS FLOAT) > 0.5 AND actual_winner = 'home' THEN false
    -- Default to null if no clear prediction
    ELSE null
  END
)
WHERE game_completed = true AND actual_winner IS NOT NULL;

-- NBA predictions
UPDATE nba_predictions
SET prediction_correct = (
  CASE 
    -- If home team was predicted to win (home_ml_prob > 0.5) and home team actually won
    WHEN CAST(home_ml_prob AS FLOAT) > 0.5 AND actual_winner = 'home' THEN true
    -- If away team was predicted to win (away_ml_prob > 0.5) and away team actually won
    WHEN CAST(away_ml_prob AS FLOAT) > 0.5 AND actual_winner = 'away' THEN true
    -- If home team was predicted to win (home_ml_prob > 0.5) but away team won
    WHEN CAST(home_ml_prob AS FLOAT) > 0.5 AND actual_winner = 'away' THEN false
    -- If away team was predicted to win (away_ml_prob > 0.5) but home team won
    WHEN CAST(away_ml_prob AS FLOAT) > 0.5 AND actual_winner = 'home' THEN false
    -- Default to null if no clear prediction
    ELSE null
  END
)
WHERE game_completed = true AND actual_winner IS NOT NULL;

-- Show summary of changes
SELECT 
  'NFL' as sport,
  COUNT(*) as total_completed,
  SUM(CASE WHEN prediction_correct = true THEN 1 ELSE 0 END) as correct_predictions,
  SUM(CASE WHEN prediction_correct = false THEN 1 ELSE 0 END) as incorrect_predictions,
  ROUND(100.0 * SUM(CASE WHEN prediction_correct = true THEN 1 ELSE 0 END) / COUNT(*), 2) as accuracy_pct
FROM nfl_model_predictions
WHERE game_completed = true AND actual_winner IS NOT NULL

UNION ALL

SELECT 
  'NBA' as sport,
  COUNT(*) as total_completed,
  SUM(CASE WHEN prediction_correct = true THEN 1 ELSE 0 END) as correct_predictions,
  SUM(CASE WHEN prediction_correct = false THEN 1 ELSE 0 END) as incorrect_predictions,
  ROUND(100.0 * SUM(CASE WHEN prediction_correct = true THEN 1 ELSE 0 END) / COUNT(*), 2) as accuracy_pct
FROM nba_predictions
WHERE game_completed = true AND actual_winner IS NOT NULL;

