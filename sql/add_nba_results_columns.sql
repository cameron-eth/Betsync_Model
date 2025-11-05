-- Add result tracking columns to nba_predictions table
-- Run this migration in Supabase SQL Editor

ALTER TABLE nba_predictions
  ADD COLUMN IF NOT EXISTS actual_home_score INT,
  ADD COLUMN IF NOT EXISTS actual_away_score INT,
  ADD COLUMN IF NOT EXISTS actual_winner TEXT CHECK (actual_winner IN ('home', 'away', NULL)),
  ADD COLUMN IF NOT EXISTS game_completed BOOLEAN DEFAULT false,
  ADD COLUMN IF NOT EXISTS result_synced_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS prediction_correct BOOLEAN;

-- Add index for querying incomplete games
CREATE INDEX IF NOT EXISTS idx_nba_predictions_incomplete 
  ON nba_predictions(game_date, game_completed) 
  WHERE game_completed = false;

-- Add index for analytics queries
CREATE INDEX IF NOT EXISTS idx_nba_predictions_completed 
  ON nba_predictions(game_completed, confidence_label) 
  WHERE game_completed = true;

-- Add comments explaining the schema
COMMENT ON COLUMN nba_predictions.actual_home_score IS 'Final score for home team';
COMMENT ON COLUMN nba_predictions.actual_away_score IS 'Final score for away team';
COMMENT ON COLUMN nba_predictions.actual_winner IS 'Who won: home or away';
COMMENT ON COLUMN nba_predictions.game_completed IS 'Whether the game has finished and results are synced';
COMMENT ON COLUMN nba_predictions.result_synced_at IS 'When the actual results were synced';
COMMENT ON COLUMN nba_predictions.prediction_correct IS 'Whether we predicted the winner correctly';

