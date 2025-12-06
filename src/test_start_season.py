
import pandas as pd
from model_simple_rolling import Model
from environment import Environment

def test_start_season():
    print("Testing start_season parameter...")
    
    # Create fake games data
    # Seasons 2007, 2008, 2009
    games_data = []
    import datetime
    base_date = datetime.datetime(2007, 10, 1)
    
    for season in [2007, 2008, 2009, 2010]:
        for i in range(10): # 10 games per season
            game = {
                'Season': season,
                'Date': base_date + datetime.timedelta(days=i + (season-2007)*365),
                'Open': base_date + datetime.timedelta(days=i + (season-2007)*365 - 1),
                'HID': 1, 'AID': 2,
                'HS': 3 if i % 2 == 0 else 1, 'AS': 2 if i % 2 == 0 else 4,
                'H': True if i % 2 == 0 else False, 'A': False if i % 2 == 0 else True, 'D': False,
                'OddsH': 2.0, 'OddsA': 2.0, 'OddsD': 3.0,
                'Min_bet': 5, 'Max_bet': 100, 'Bankroll': 1000
            }
            # Add dummy stats
            missing_cols = ['H_SOG', 'A_SOG', 'H_PPG', 'A_PPG', 'H_PEN', 'A_PEN', 'H_FO', 'A_FO', 'Special',
                          'H_MAJ', 'A_MAJ', 'H_SHG', 'A_SHG', 'H_SV', 'A_SV', 'H_PIM', 'A_PIM', 
                          'H_BLK_S', 'A_BLK_S', 'H_HIT', 'A_HIT', 'H_BLK', 'A_BLK', 
                          'H_P1', 'A_P1', 'H_P2', 'A_P2', 'H_P3', 'A_P3', 'H_OT', 'A_OT', 'H_SO', 'A_SO']
            for stat in missing_cols:
                game[stat] = 0
            game['Special'] = ''
            games_data.append(game)
            
    df = pd.DataFrame(games_data)
    
    # Test 1: No start season (Default)
    print("\n--- Test 1: No start season ---")
    model1 = Model(required_games=5, retrain_freq=1) # Reduced required_games and retrain_freq
    env1 = Environment(df, model1, init_bankroll=1000, min_bet=5, max_bet=100)
    env1.run()
    print(f"Predictions made (No filter): {model1.predictions_made}")
    # Should predict all games after required_games
    # Total games 40. Required 5. Should predict ~35.
    
    # Test 2: Start season 2009
    print("\n--- Test 2: Start season 2009 ---")
    model2 = Model(required_games=5, retrain_freq=1, start_season=2009)
    env2 = Environment(df, model2, init_bankroll=1000, min_bet=5, max_bet=100)
    env2.run()
    print(f"Predictions made (Start 2009): {model2.predictions_made}")
    
    # Games in 2007, 2008 should be skipped.
    # Games in 2009 start processing.
    # Training data accumulation starts at 2009.
    # First 5 games of 2009 are used for training (required_games=5).
    # Predictions start after 5 games in 2009.
    # 2009 has 10 games -> 5 training, 5 predictions.
    # 2010 has 10 games -> 10 predictions.
    # Total ~15 predictions.
    
    assert model1.predictions_made > model2.predictions_made, "With filter, should make fewer predictions"
    print("✅ Model made fewer predictions with start_season set.")
    
    # Verify training data content
    # In model2, training data should only contain games from 2009+?
    # Actually Model doesn't store season in training_data (it stores features + outcome).
    # But count should be consistent.
    print(f"Training samples (Model 1): {len(model1.training_data)}")
    print(f"Training samples (Model 2): {len(model2.training_data)}")
    
    if len(model2.training_data) < len(model1.training_data):
        print("✅ Training data filtered correctly.")
    else:
        print("❌ Training data count mismatch!")

if __name__ == "__main__":
    test_start_season()
