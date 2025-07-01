
import json
import re
from typing import Dict, Any, Optional

class LLMGameActionParser:
    """
    Parses natural language commands from the LLM into structured game actions
    that can be executed by the TowerDefenseEnv.
    """
    def __init__(self):
        pass

    def parse_action(self, llm_text: str) -> Optional[Dict[str, Any]]:
        """
        Attempts to parse a game action from the LLM's text output.
        Returns a dictionary representing the action or None if parsing fails.
        """
        llm_text = llm_text.lower()

        # 1. Place Tower Action
        match_place_tower = re.search(r'place\s+(.+?)\s+tower\s+at\s+\(?(\d+),\s*(\d+)\)?', llm_text)
        if match_place_tower:
            tower_type = match_place_tower.group(1).strip()
            x = int(match_place_tower.group(2))
            y = int(match_place_tower.group(3))
            return {"type": "place_tower", "x": x, "y": y, "tower_type": tower_type}

        # 2. Start Wave Action
        match_start_wave = re.search(r'start\s+wave(?:\s+with\s+(\d+)\s+enemies)?', llm_text)
        if match_start_wave:
            num_enemies = int(match_start_wave.group(1)) if match_start_wave.group(1) else 5 # Default to 5 enemies
            return {"type": "start_wave", "num_enemies": num_enemies}

        # 3. Upgrade Tower Action (example)
        match_upgrade_tower = re.search(r'upgrade\s+tower\s+at\s+\(?(\d+),\s*(\d+)\)?', llm_text)
        if match_upgrade_tower:
            x = int(match_upgrade_tower.group(1))
            y = int(match_upgrade_tower.group(2))
            return {"type": "upgrade_tower", "x": x, "y": y}

        # 4. No-op or unknown action
        if "no operation" in llm_text or "do nothing" in llm_text:
            return {"type": "no_op"}

        print(f"Could not parse action from LLM text: {llm_text}")
        return None

# Example Usage:
# if __name__ == "__main__":
#     parser = LLMGameActionParser()

#     # Test cases
#     print(parser.parse_action("Please place a basic tower at (2,3)."))
#     print(parser.parse_action("Start wave with 10 enemies."))
#     print(parser.parse_action("Upgrade the tower at (5,5)."))
#     print(parser.parse_action("Just do nothing for now."))
#     print(parser.parse_action("Move enemy to (1,1).")) # Should return None


