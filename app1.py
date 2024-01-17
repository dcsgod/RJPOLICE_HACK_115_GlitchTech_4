from flask import Flask, render_template
import sqlite3
import json

app = Flask(__name__)

@app.route('/')
def index():
    # Read section number from JSON file
    with open('firbreak.json', 'r') as json_file:
        data = json.load(json_file)
        section_number = data.get('section_number')

    # Get section information
    section_info = get_section_info(section_number)

    return render_template('index.html', section_info=section_info)

def get_section_info(section_number):
    # Connect to your SQLite database
    conn = sqlite3.connect('rph')
    cursor = conn.cursor()

    # Execute the SQL query
    cursor.execute(f"SELECT Offense, Bailable, Cognizable, Court FROM mytable WHERE URL LIKE '%/section-{section_number}%'")

    # Fetch the result
    result = cursor.fetchone()

    # Close the database connection
    conn.close()

    return result

if __name__ == '__main__':
    app.run(debug=True)
