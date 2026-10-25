
You are an expert in knowledge-graph construction.

Your task:
Given a pair of (Title, Content), where **Title is the central entity of this text segment**, your goal is to:

1) Extract entities that are **directly related to the Title**
2) Extract relationships that **explicitly involve the Title**
3) Format output EXACTLY as specified
--------------------------
### CORE ASSUMPTION
--------------------------
- The Title represents the **anchor / central entity** of this Content
- Entity and relationship extraction MUST be centered around the Title
- Do NOT extract entities or relationships that are not semantically connected to the Title
- **Extract as many entities and relationships as possible without violating the above rules**


--------------------------
### ENTITY EXTRACTION RULE
--------------------------
Each entity MUST be formatted as:

("entity"||entity_name||entity_type||description)

Rules:
- The Title entity MUST be extracted first
- description = The EXACT sentence(s) from the Content that mention or describe this entity.The extracted sentences must have a complete subject-verb-object structure, with both the subject and the object being meaningful nouns.No pronouns shall be used.
- description: No more than four sentences in total.
- Allowed entity types = {entity_types} (preferred but not mandatory)
- Professional nouns shall not be split.


--------------------------
### RELATIONSHIP EXTRACTION RULE
--------------------------
Each relationship MUST be formatted as:

("relationship"||source_entity||target_entity||description||keyword||strength)

### Rules:
- At least one of (source_entity, target_entity) MUST be the Title
- description = The EXACT sentence(s) from the Content that provide the evidence for this relationship. 
- keyword MUST be a **specific and semantically precise verb or verb phrase**
    BAD: "relate to", "is", "connected with", "involved in".
    GOOD: "represented", "conceived_by", "defeated", "administrative_center_of", "daughter_of".
- DO NOT infer or invent relationships not explicitly supported by Content
--------------------------

strength RULES:
- Integer between 1–10

--------------------------
### CONSTRAINTS
--------------------------
- DO NOT add extra fields
- Output ALL entities first, then ALL relationships
- Every entity referenced in a relationship MUST appear in entity list
- If unsure about entity type, choose the closest reasonable type
- DO NOT invent relationships not supported by the text
- Pronouns shall not appear in the subject and object of description

--------------------------
### TEXT
--------------------------
{input_text}
--------------------------
### EXAMPLE
--------------------------

Title: Niassa Reserve
Content: Niassa Reserve is a nature reserve in Cabo Delgado Province and Niassa Province, Mozambique. Covering over , it is the largest protected area in the country. The reserve is part of the Trans-Frontier Conservation Area and links to the Tanzanian Lukwika-Lumesule Game Reserve. It will connect to the Lake Niassa Reserve when it is completed.

Title: Michurinsky District
Content: Michurinsky District () is an administrative and municipal district (raion), one of the twenty-three in Tambov Oblast, Russia. It is located in the northwest of the oblast. The district borders with Pervomaysky District in the north, Nikiforovsky District in the east, Petrovsky District in the south, and with Dobrovsky District of Lipetsk Oblast in the west. The area of the district is . Its administrative center is the town of Michurinsk (which is not administratively a part of the district). Population: 34,245 (2010 Census);

Title: Shmidtovsky District
Content: Shmidtovsky District () was an administrative district (raion) of Chukotka Autonomous Okrug, Russia, which existed in 1973–2011. As a municipal division, it was, together with Iultinsky Administrative District, incorporated as Iultinsky Municipal District. It was located in the far north of the autonomous okrug and bordered with Iultinsky District in the east, Anadyrsky District in the south, and with Chaunsky District in the west. The area of the district was . Its administrative center was the urban locality (an urban-type settlement) of Mys Shmidta. Population: The population of Mys Shmidta accounted for 33.5% of the district's total population.

--------------------------
### Output format:
- Each record (entity or relationship) MUST end with: {record_delimiter}
- Final output MUST end with: {completion_delimiter}

Example entity:
("entity"||"Niassa Reserve"||"location"||"Niassa Reserve is a nature reserve in Cabo Delgado Province and Niassa Province, Mozambique."){record_delimiter} 
("entity"||"Mozambique"||"location"||"Niassa Reserve is a nature reserve in Cabo Delgado Province and Niassa Province, Mozambique."){record_delimiter} 
("entity"||"Lukwika-Lumesule Game Reserve"||"location"||"The reserve is part of the Trans-Frontier Conservation Area and links to the Tanzanian Lukwika-Lumesule Game Reserve."){record_delimiter} 
("entity"||"Michurinsky District"||"administrative_district"||"Michurinsky District () is an administrative and municipal district (raion), one of the twenty-three in Tambov Oblast, Russia."){record_delimiter} 
("entity"||"Tambov Oblast"||"location"||"Michurinsky District () is an administrative and municipal district (raion), one of the twenty-three in Tambov Oblast, Russia."){record_delimiter} 
("entity"||"Michurinsk"||"location"||"Its administrative center is the town of Michurinsk (which is not administratively a part of the district)."){record_delimiter} 
("entity"||"Shmidtovsky District"||"administrative_district"||"Shmidtovsky District () was an administrative district (raion) of Chukotka Autonomous Okrug, Russia, which existed in 1973–2011."){record_delimiter} 
("entity"||"Chukotka Autonomous Okrug"||"location"||"Shmidtovsky District () was an administrative district (raion) of Chukotka Autonomous Okrug, Russia, which existed in 1973–2011."){record_delimiter} 
("entity"||"Iultinsky Municipal District"||"administrative_district"||"As a municipal division, it was, together with Iultinsky Administrative District, incorporated as Iultinsky Municipal District."){record_delimiter} 
("entity"||"Mys Shmidta"||"location"||"Its administrative center was the urban locality (an urban-type settlement) of Mys Shmidta."){record_delimiter}
Example relationship:
("relationship"||"Niassa Reserve"||"Mozambique"||"Niassa Reserve is a nature reserve in Cabo Delgado Province and Niassa Province, Mozambique."||"located_in"||10){record_delimiter} 
("relationship"||"Niassa Reserve"||"Lukwika-Lumesule Game Reserve"||"The reserve is part of the Trans-Frontier Conservation Area and links to the Tanzanian Lukwika-Lumesule Game Reserve."||"links_to"||8){record_delimiter} 
("relationship"||"Niassa Reserve"||"Lake Niassa Reserve"||"It will connect to the Lake Niassa Reserve when it is completed."||"will_connect_to"||7){record_delimiter} 
("relationship"||"Michurinsky District"||"Tambov Oblast"||"Michurinsky District () is an administrative and municipal district (raion), one of the twenty-three in Tambov Oblast, Russia."||"part_of"||10){record_delimiter} 
("relationship"||"Michurinsky District"||"Michurinsk"||"Its administrative center is the town of Michurinsk (which is not administratively a part of the district)."||"administrative_center"||10){record_delimiter} 
("relationship"||"Michurinsky District"||"Pervomaysky District"||"The district borders with Pervomaysky District in the north..."||"borders"||8){record_delimiter} 
("relationship"||"Shmidtovsky District"||"Chukotka Autonomous Okrug"||"Shmidtovsky District () was an administrative district (raion) of Chukotka Autonomous Okrug, Russia..."||"part_of"||10){record_delimiter} 
("relationship"||"Shmidtovsky District"||"Iultinsky Municipal District"||"As a municipal division, it was, together with Iultinsky Administrative District, incorporated as Iultinsky Municipal District."||"incorporated_into"||10){record_delimiter} 
("relationship"||"Shmidtovsky District"||"Mys Shmidta"||"Its administrative center was the urban locality (an urban-type settlement) of Mys Shmidta."||"administrative_center"||10){record_delimiter} 
("relationship"||"Shmidtovsky District"||"Chaunsky District"||"It was located in the far north of the autonomous okrug and bordered with... and with Chaunsky District in the west."||"bordered"||8){record_delimiter} 
{completion_delimiter}
