-- Add columns to track actual game outcomes for performance tracking

-- NFL predictions table
ALTER TABLE nfl_model_predictions
ADD COLUMN IF NOT EXISTS actual_winner TEXT,
ADD COLUMN IF NOT EXISTS home_score INTEGER,
ADD COLUMN IF NOT EXISTS away_score INTEGER,
ADD COLUMN IF NOT EXISTS game_completed BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS completed_at TIMESTAMP WITH TIME ZONE;

-- NBA predictions table
ALTER TABLE nba_predictions
ADD COLUMN IF NOT EXISTS actual_winner TEXT,
ADD COLUMN IF NOT EXISTS home_score INTEGER,
ADD COLUMN IF NOT EXISTS away_score INTEGER,
ADD COLUMN IF NOT EXISTS game_completed BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS completed_at TIMESTAMP WITH TIME ZONE;

-- Create indexes for faster performance queries
CREATE INDEX IF NOT EXISTS idx_nfl_actual_winner ON nfl_model_predictions(actual_winner) WHERE actual_winner IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_nfl_game_completed ON nfl_model_predictions(game_completed, game_date DESC);

CREATE INDEX IF NOT EXISTS idx_nba_actual_winner ON nba_predictions(actual_winner) WHERE actual_winner IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_nba_game_completed ON nba_predictions(game_completed, game_date DESC);

-- Add comments
COMMENT ON COLUMN nfl_model_predictions.actual_winner IS 'The team that actually won the game (for hit rate tracking)';
COMMENT ON COLUMN nfl_model_predictions.home_score IS 'Final home team score';
COMMENT ON COLUMN nfl_model_predictions.away_score IS 'Final away team score';
COMMENT ON COLUMN nfl_model_predictions.game_completed IS 'Whether the game has been played';
COMMENT ON COLUMN nfl_model_predictions.completed_at IS 'When the game result was recorded';

COMMENT ON COLUMN nba_predictions.actual_winner IS 'The team that actually won the game (for hit rate tracking)';
COMMENT ON COLUMN nba_predictions.home_score IS 'Final home team score';
COMMENT ON COLUMN nba_predictions.away_score IS 'Final away team score';
COMMENT ON COLUMN nba_predictions.game_completed IS 'Whether the game has been played';
COMMENT ON COLUMN nba_predictions.completed_at IS 'When the game result was recorded';

