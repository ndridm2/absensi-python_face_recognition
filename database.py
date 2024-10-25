import mysql.connector
import uuid

connection = mysql.connector.connect(
    user='root',
    password='pwdpwd8',
    host='127.0.0.1',
    database='absensi-db',
    collation='utf8mb4_unicode_ci',
    raise_on_warnings=True
)

if connection.is_connected():
    print("Connected")
else:
    print("Not Connected")

def fetchUser(userId):
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE id = %s", (userId,))
    return cursor.fetchone()
def storeAttendance(userId):
    cursor = connection.cursor()
    query = "INSERT INTO attendances (id, user_id, status, description) VALUES (%s, %s, %s, %s)"
    values = (str(uuid.uuid4()), userId, 'attend', 'face recognition')
    cursor.execute(query, values)
    connection.commit()

# user = fetchUser(1)
# print(user['name']) 