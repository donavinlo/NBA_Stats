# NBA Stats
For the past 6 months I have been teaching myself the intuition behind Data Science. For this project, I decided to apply the basics of machine learning, data cleaning, data pre-processing, data analysis, and model evaluation. 

This project provides a model to attribute the most important statistcal features that correspond a win in the National Basketball Association. This project was meant to be a quick learning experience. There are many strategies I could of implemented to create better insight and develop a better model. At the end of this introduction I will discuss a couple strategies I will implement if I did this project again.

### Packages Used
- Numpy
- Matplotlib
- Scikitlearn
- Seaborn
- os

### File Explanation
*NBAstats.csv* contains all the features used for this simple datascience project. All statistics within a basketball for all NBA teams from 2014 to 2018

### Feature Definition
- Team: A team within the NBA 
- Game: The game number that team has played that year (1-82)
- Location: Whether the team played at Home (0: Home, 1: Away)
- Opponent: Who the team's opponent was
- WINorLOSS: If the team won the game (0 for loss, 1 for win)
- TeamPoints: How many points the team has scored
-OpponentPoints: Opponent Points Scored
- FG: Field Goals made
- FGA: Field Goals Attempted
- FG%: Field Goal Percentage
- 3ptShots: 3-point shots made
- 3ptShotsAtt: 3-point shots attempted
- 3ptShot%: 3-point shot %
- FT: Free Throws MAde
- FTA: Free Throws Attempted
- FT%: Free Throw %
- AST: Assists
- STL: Team steals
- OREB: Offensive Rebounds
- REB: Total Rebound
- BLK: Team Blocks
- TO: Turnovers
- Month: The month the game was played
- Year: The year the game was played 
- TotalFouls: Team total fouls for the game 

### Reflections
