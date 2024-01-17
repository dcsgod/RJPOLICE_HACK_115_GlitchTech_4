from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        section_number = request.form['section_number']
        section_info = get_section_info(section_number)
        return render_template('index.html', section_info=section_info)
    return render_template('index.html', section_info=None)

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
