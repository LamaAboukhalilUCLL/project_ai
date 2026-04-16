import xml.etree.ElementTree as ET
import psycopg2

conn = psycopg2.connect(
    dbname="stackexchange_db",
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

print("Loading users...")
tree = ET.parse("Users.xml")
for row in tree.getroot():
    a = row.attrib
    try:
        cur.execute(
            "INSERT INTO users VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
            (a.get("Id"), a.get("DisplayName"), a.get("Location"),
             a.get("Reputation", 0), a.get("CreationDate"))
        )
    except:
        conn.rollback()
        continue
conn.commit()
print("Users done.")

print("Loading posts...")
tree = ET.parse("Posts.xml")
for row in tree.getroot():
    a = row.attrib
    try:
        cur.execute(
            "INSERT INTO posts VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
            (a.get("Id"), a.get("PostTypeId"), a.get("OwnerUserId"),
             a.get("Title"), a.get("Score", 0),
             a.get("ViewCount", 0), a.get("AnswerCount", 0), a.get("CreationDate"))
        )
    except:
        conn.rollback()
        continue
conn.commit()
print("Posts done.")

print("Loading votes...")
tree = ET.parse("Votes.xml")
for row in tree.getroot():
    a = row.attrib
    try:
        cur.execute(
            "INSERT INTO votes VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING",
            (a.get("Id"), a.get("PostId"), a.get("VoteTypeId"), a.get("CreationDate"))
        )
    except:
        conn.rollback()
        continue
conn.commit()
print("Votes done.")

cur.close()
conn.close()
print("All done!")