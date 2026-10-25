# Entity Extraction Prompt Examples

## Example 1

Entity_types: ["person","organization", "Location", "Date", "Number","Artwork","Adjective","Event","Theory"]

Text:
'''
"context":[["Nancy Sinatra",["Nancy Sandra Sinatra (born June 8, 1940) is an American singer and actress."," She is the elder daughter of Frank Sinatra and Nancy (Barbato) Sinatra, and is widely known for her 1966 signature hit \"These Boots Are Made for Walkin'\"."]],["Kon Kan",["Kon Kan is a Canadian synthpop project conceived and formed in 1988 by Barry Harris in Toronto, Ontario."," Kon Kan were awarded a 1990 Juno for the song \"I Beg Your Pardon\" and nominated for a 1991 Juno Award for the single \"Puss N' Boots\/These Boots Are Made for Walkin'\"."]]]
'''

Output:
("entity"||"Nancy Sinatra"||"person"||"Nancy Sandra Sinatra (born June 8, 1940) is an American singer and actress."){record_delimiter} 
("entity"||"Frank Sinatra"||"person"||"She is the elder daughter of Frank Sinatra and Nancy (Barbato) Sinatra."){record_delimiter} 
("entity"||"Nancy (Barbato) Sinatra"||"person"||"She is the elder daughter of Frank Sinatra and Nancy (Barbato) Sinatra."){record_delimiter} 
("entity"||"These Boots Are Made for Walkin'"||"creative_work"||"She is widely known for her 1966 signature hit \"These Boots Are Made for Walkin'\"."){record_delimiter} 
("entity"||"Kon Kan"||"organization"||"Kon Kan is a Canadian synthpop project conceived and formed in 1988 by Barry Harris in Toronto, Ontario."){record_delimiter} 
("entity"||"Barry Harris"||"person"||"Kon Kan is a Canadian synthpop project conceived and formed in 1988 by Barry Harris in Toronto, Ontario."){record_delimiter} 
("entity"||"Toronto"||"location"||"Kon Kan is a Canadian synthpop project conceived and formed in 1988 by Barry Harris in Toronto, Ontario."){record_delimiter} 
("entity"||"I Beg Your Pardon"||"creative_work"||"Kon Kan were awarded a 1990 Juno for the song \"I Beg Your Pardon\"."){record_delimiter} 
("entity"||"Puss N' Boots/These Boots Are Made for Walkin'"||"creative_work"||"Kon Kan were nominated for a 1991 Juno Award for the single \"Puss N' Boots/These Boots Are Made for Walkin'\"."){record_delimiter} 
("entity"||"Juno Award"||"award"||"Kon Kan were awarded a 1990 Juno for the song \"I Beg Your Pardon\" and nominated for a 1991 Juno Award for the single."){record_delimiter}

("relationship"||"Nancy Sinatra"||"Frank Sinatra"||"She is the elder daughter of Frank Sinatra and Nancy (Barbato) Sinatra."||"daughter_of"||10){record_delimiter} 
("relationship"||"Nancy Sinatra"||"Nancy (Barbato) Sinatra"||"She is the elder daughter of Frank Sinatra and Nancy (Barbato) Sinatra."||"daughter_of"||10){record_delimiter} 
("relationship"||"Nancy Sinatra"||"These Boots Are Made for Walkin'"||"She is widely known for her 1966 signature hit \"These Boots Are Made for Walkin'\"."||"performed_by"||10){record_delimiter} 
("relationship"||"Kon Kan"||"Barry Harris"||"Kon Kan is a Canadian synthpop project conceived and formed in 1988 by Barry Harris in Toronto, Ontario."||"formed_by"||10){record_delimiter} 
("relationship"||"Kon Kan"||"Toronto"||"Kon Kan is a Canadian synthpop project conceived and formed in 1988 by Barry Harris in Toronto, Ontario."||"originates_from"||9){record_delimiter} 
("relationship"||"Kon Kan"||"I Beg Your Pardon"||"Kon Kan were awarded a 1990 Juno for the song \"I Beg Your Pardon\"."||"awarded_for"||10){record_delimiter} 
("relationship"||"Kon Kan"||"Juno Award"||"Kon Kan were awarded a 1990 Juno for the song \"I Beg Your Pardon\"."||"received"||10){record_delimiter} 
("relationship"||"Kon Kan"||"Puss N' Boots/These Boots Are Made for Walkin'"||"Kon Kan were nominated for a 1991 Juno Award for the single \"Puss N' Boots/These Boots Are Made for Walkin'\"."||"nominated_for"||8){record_delimiter} 
("relationship"||"These Boots Are Made for Walkin'"||"Puss N' Boots/These Boots Are Made for Walkin'"||"nominated for a 1991 Juno Award for the single \"Puss N' Boots/These Boots Are Made for Walkin'\"."||"featured_in"||7){record_delimiter} 
{completion_delimiter}


## Example 2

Entity_types: ["person","organization", "Location", "Date", "Number","Artwork","Adjective","Event","Theory"]

Text:
'''
"context":[["2012–13 VCU Rams men's basketball team",["The 2012–13 VCU Rams men's basketball team represented Virginia Commonwealth University during the 2012–13 NCAA Division I men's basketball season."," It was the 45th season of the University fielding a men's basketball program."," Led by fourth-year head coach Shaka Smart, they played their home games at the Stuart C. Siegel Center."," This was the Rams inaugural season in the Atlantic 10 Conference (A10), after spending the past 17 years in the Colonial Athletic Association (CAA)."," They finished the season 27–9, 12–4 in A10 play to finish in second place."," They advanced to the championship game of the 2013 Atlantic 10 Tournament where they lost to Saint Louis."," They received an at-large bid to the 2013 NCAA Tournament, their third straight NCAA Tournament appearance, where they defeated Akron in the first round before losing in the third round to Michigan."]]]
'''

Output:
("entity"||"2012–13 VCU Rams men's basketball team"||"sports_team"||"The 2012–13 VCU Rams men's basketball team represented Virginia Commonwealth University during the 2012–13 NCAA Division I men's basketball season."){record_delimiter} 
("entity"||"Virginia Commonwealth University"||"organization"||"The 2012–13 VCU Rams men's basketball team represented Virginia Commonwealth University during the 2012–13 NCAA Division I men's basketball season."){record_delimiter} 
("entity"||"Shaka Smart"||"person"||"Led by fourth-year head coach Shaka Smart, they played their home games at the Stuart C. Siegel Center."){record_delimiter} 
("entity"||"Stuart C. Siegel Center"||"location"||"Led by fourth-year head coach Shaka Smart, they played their home games at the Stuart C. Siegel Center."){record_delimiter} 
("entity"||"Atlantic 10 Conference"||"organization"||"This was the Rams inaugural season in the Atlantic 10 Conference (A10), after spending the past 17 years in the Colonial Athletic Association (CAA)."){record_delimiter} 
("entity"||"Colonial Athletic Association"||"organization"||"This was the Rams inaugural season in the Atlantic 10 Conference (A10), after spending the past 17 years in the Colonial Athletic Association (CAA)."){record_delimiter} 
("entity"||"Saint Louis"||"sports_team"||"They advanced to the championship game of the 2013 Atlantic 10 Tournament where they lost to Saint Louis."){record_delimiter} 
("entity"||"2013 NCAA Tournament"||"event"||"They received an at-large bid to the 2013 NCAA Tournament, their third straight NCAA Tournament appearance."){record_delimiter} 
("entity"||"Akron"||"sports_team"||"where they defeated Akron in the first round before losing in the third round to Michigan."){record_delimiter} 
("entity"||"Michigan"||"sports_team"||"where they defeated Akron in the first round before losing in the third round to Michigan."){record_delimiter}

("relationship"||"2012–13 VCU Rams men's basketball team"||"Virginia Commonwealth University"||"The 2012–13 VCU Rams men's basketball team represented Virginia Commonwealth University during the 2012–13 NCAA Division I men's basketball season."||"represented"||10){record_delimiter} 
("relationship"||"2012–13 VCU Rams men's basketball team"||"Shaka Smart"||"Led by fourth-year head coach Shaka Smart, they played their home games at the Stuart C. Siegel Center."||"coached_by"||10){record_delimiter} 
("relationship"||"2012–13 VCU Rams men's basketball team"||"Stuart C. Siegel Center"||"Led by fourth-year head coach Shaka Smart, they played their home games at the Stuart C. Siegel Center."||"home_arena"||9){record_delimiter} 
("relationship"||"2012–13 VCU Rams men's basketball team"||"Atlantic 10 Conference"||"This was the Rams inaugural season in the Atlantic 10 Conference (A10), after spending the past 17 years in the Colonial Athletic Association (CAA)."||"member_of"||10){record_delimiter} 
("relationship"||"2012–13 VCU Rams men's basketball team"||"Colonial Athletic Association"||"This was the Rams inaugural season in the Atlantic 10 Conference (A10), after spending the past 17 years in the Colonial Athletic Association (CAA)."||"formerly_member_of"||8){record_delimiter} 
("relationship"||"2012–13 VCU Rams men's basketball team"||"Saint Louis"||"They advanced to the championship game of the 2013 Atlantic 10 Tournament where they lost to Saint Louis."||"lost_to"||8){record_delimiter} 
("relationship"||"2012–13 VCU Rams men's basketball team"||"2013 NCAA Tournament"||"They received an at-large bid to the 2013 NCAA Tournament, their third straight NCAA Tournament appearance."||"appeared_in"||9){record_delimiter} 
("relationship"||"2012–13 VCU Rams men's basketball team"||"Akron"||"where they defeated Akron in the first round before losing in the third round to Michigan."||"defeated"||8){record_delimiter} 
("relationship"||"2012–13 VCU Rams men's basketball team"||"Michigan"||"where they defeated Akron in the first round before losing in the third round to Michigan."||"lost_to"||8){record_delimiter} 
("relationship"||"Shaka Smart"||"2012–13 VCU Rams men's basketball team"||"Led by fourth-year head coach Shaka Smart, they played their home games..."||"led"||10){record_delimiter} 
{completion_delimiter}