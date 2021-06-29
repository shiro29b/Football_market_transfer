# Football transfer analysis
understanding football transfer from 2003-2004 season to 2018-2019 season and creating a ML model to predict the market values of future players
some important things to consider before we explore the dataset:
* dataset taken from : https://www.kaggle.com/vardan95ghazaryan/top-250-football-transfers-from-2000-to-2018?select=top250-00-19.csv
* the code is in python 3.9 and for visual representation i have used upyter notebook
* however, this is not the final code, will add more features to make the model more accurate 

# Overview of the trend
Football clubs these days are paying exorbitant prices to get their players. Defenders are going for €60-70 million, playmakers for €100-120 million and superstars for even more. Every club and fan will try and justify the amount spent on certain players and the points they make might be valid at the end. But all in all, paying millions in the 60s and 70s is too much and it's hurting the market in a bad way.

Harry Maguire might be the answer to your defensive problems, but paying €93 million for a player who's shown only glimpses of skill and has been good only over two seasons is just throwing money. But to be fair to Harry, he needs time to settle and United aren't in a good place right now. Maybe if the things turn around for them, we can see if Maguire is worth the money.

As the years go by, we can only expect the trend to rise - clubs will only go out of their way and we can see them paying more and more in the hopes of getting that one player who’ll take them to great heights.

Rumours of PSG wonderkid, Kylian Mbappé, having a price tag of €500 million isn’t far from the truth. And only one club is even remotely considering to buy him at that price.

# about the dataset
here is a screenshot about the dataset
![Screenshot from 2021-06-29 14-53-27](https://user-images.githubusercontent.com/53612010/123773004-f12a6880-d8e9-11eb-9a70-b05afecff448.png)

It contains:
* Name of the Players
* Primary Position they play
* Age of the Player
* The club they've been bought from
* The league they've been bought from
* The club that bought them
* The league their new club belongs to
* The season in which the transfer took place
* Market Value of the player
* Transfer Fee paid for the player

# Looking at some features
we calculated the difference of transfer fee and market value and averaged it across all the seasons and observed this trend
![Screenshot from 2021-06-29 15-02-03](https://user-images.githubusercontent.com/53612010/123774543-22576880-d8eb-11eb-9e90-d206e7f81b36.png)

we had to remove some of the datas which were Nan market value

* What does Difference tell us?

It's simple enough to see what Difference is; it's just the difference between the Transfer Fee and Market Value. But in a footballing context, it's a bit more elaborate. As you know, a club sets a market value for a player based on various factors(age, position, birth country, skill, etc), and a buying club almost never pays the exact amount set by the selling club. It's sometimes(mostly) more than the market value and sometimes less.

when we plot the trend we observe this

![Screenshot from 2021-06-29 15-07-39](https://user-images.githubusercontent.com/53612010/123775278-de189800-d8eb-11eb-8492-0eb9f6fcc7d5.png)

we can see a decrease from ten thousand euros to an abysmal negative €3 million. The negative value is an indication of football clubs paying less than the market value for players in the 2006/2007 season. This means that on an average, players were being undersold.

This phenomenon can be attributed to the 2006 Italian Football Scandal, a.k.a Calciopoli.

Towards the end of the 2005/2006 season, many Italian clubs were caught in a match-fixing scandal. Investigations discovered that teams paid money to get favourable referees for their matches helping them win games.

Back in 2011, the State of Qatar acquired Paris Saint-Germain F.C.(PSG) through Qatar Sports Investments(QSI). But initially, the new owners didn't spend money. During and after the 2012/2013, PSG hit the market buying superstars like Zlatan Ibrahimović, Edinson Cavani, Thiago Silva and other talented players like Lavezzi, Marco Verratti, Lucas Moura and Digne.

Contrary to popular belief, PSG didn't splash huge amounts of money on these players. For the majority of their transfers, they paid only a fraction above the market value, they acquired Ibrahimović for less than 43% of his market value. Unlike today, PSG at the beginning of the takeover did good business and made many sensible buys ushering in their era of dominance over the Ligue One.

We can see a spike from negative ten thousand euros to a staggering €1.6 million in the season 2013/2014 and the reason may lie in one transfer; Gareth Bale moving from Tottenham Hotspurs to Real Madrid for €101 million. The Los Blancos paid 55% more than the market value for Bale. In my personal opinion, this I believe is where the trend of overpaying for remotely good players in the hopes that they'd become superstars began.

Today's transfer market is bloated and it will only become worse as the seasons go by. And this can be seen in the meteoric rise the Average Difference makes from the 2014/2015 season to the 2017/2018 season; from €1.6 million to €7 million. This means that the average amount of money football clubs are paying for players above the market was €7 million as of 2017/2018. That is a massive 337% increase in overspending. If this trend is to continue, which I'm sure it will, we could see huge amounts of money being paid by clubs to get talent.

Just to paint a clearer picture, here are a few transfers with huge overspending percentages in the five seasons:

* Neymar - €222 million, 122% more than market value
* Virgil Van Dijk - €78.8 million, 162% more than market value
* Paul Pogba - €105 million, 50% more than the market value
* Ousmane Dembele - €115 million, 248% more than the market value
* Anthony Martial - €60 million, 650% more than the market value

# Transfer trend with Age

![Screenshot from 2021-06-29 15-14-25](https://user-images.githubusercontent.com/53612010/123776360-d9081880-d8ec-11eb-9732-bf8817feff95.png)

it has always been observed that players with aged 20-25 are priced well in transfer market. Young, skilled players always carry a huge valuation with them and these days, that can led to hype and high transfer fees. 

# Transfer trend with position

![Screenshot from 2021-06-29 15-19-51](https://user-images.githubusercontent.com/53612010/123777193-91ce5780-d8ed-11eb-9058-ea08dde8c10b.png)

this graph depicts the number of signing of specific position over the last 15 years, clearly it is evident that the number of forward signing has outnumbered other significant signings. The demand for perfect CB is the need for perfect showstopping and controlling the opposition attack, therefore, it is with no surprise that the number of CB signings have grown rapidly over the years, with modern day full backs, it is expected that full back signings will also increase reapidly

![Screenshot from 2021-06-29 15-24-30](https://user-images.githubusercontent.com/53612010/123777835-351f6c80-d8ee-11eb-91ab-4337d1c3c59b.png)

this graph depicts the average amount of money spend on each position, it is clearly observable that pacy winger have always been sky high priced. Along with the other midfielders, the winger is actively involved in the game, providing balance when the team moves up the pitch or helping to break up the other team's play. The winger is the link between attack and defence, playing an essential role in keeping the team as a unit.

They can affect the team's attacking and defensive play in different ways depending on the phase of play, the match context and the manager's tactics. The winger may play the role of attacking winger and/or wing-back in a single match. They're known for their versatility and ability to run up and down the line.

# Transfer trend with top 5 leagues

however, I have visually represented only top5 leagues but the dataset contains more than 40 different leagues and the model is trained on all of the leaugues that have been included

![Screenshot from 2021-06-29 15-28-29](https://user-images.githubusercontent.com/53612010/123778815-22596780-d8ef-11eb-838f-ed654baffe90.png)

this graph depicts about the average market value of a player plaing in the current league

![Screenshot from 2021-06-29 15-34-01](https://user-images.githubusercontent.com/53612010/123779206-8e3bd000-d8ef-11eb-81f9-11442306122d.png)

this graph depicts the expenditure of top 5 leagues

It is clearly evident from the previous two graphs that laliga are more interested in buying players across the globe for huge amount of money, this is mostly due to Los Blancos investing hugely on players such as Figo, Bechkham, Ronaldo, Bale and many more, however, the spanish giants atletico and barcelona are into transfer market as well in the last decade. 

It can also be seen that bundesliga are interested in signing cheap players it mostly due to bayern munich heavily depending upon Borrusia Dortmund on their transfer target and on other domestic club as well. It is also noted that League 1 players have relatively low market value

# Training the data

the model is trained on KNN classification and can be seen on the py file or ipynb file

# conclusion

Why is the model inaccurate?

* Poor (or) Not Enough Features: As of the first version of this project, I haven't added multiple features in our model. Remember when I said, that the Market Value of a football player is determined by his age, birth country, position, skill and so on. And since the only dependant feature we took here is Transfer Fee, it isn't sufficient.
Note: In the future, I will add more features, try to vectorize the non-integer values and try and create an accurate model.

* The Ever-changing Game: Football happens to be the most popular sport and that makes it an ever-changing sport. For a few years, midfielders are important and then all of a sudden, defenders are the most sought after. One year, Spain is producing the best young talent, another year, Dutch youngsters are setting the stage on fire! Even if we managed to have a perfect working model right now, it would become obsolete in a few seasons.
