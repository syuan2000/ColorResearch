from color import *
import mysql.connector

try:
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        #yourRootPassword
        passwd="syuan2000",
        database="mauruemc_color"
        )
    
    movies=[("Inception", 2010, "Christopher Nolan", "Wally Pfister", "2.39 : 1"),
            ("Black Swan", 2010, "Darren Aronofsky", "Matthew Libatique", "2.35 : 1"),
            ("The King's Speech", 2010, "Tom Hooper", "Danny Cohen", "1.85 : 1"),
            ("The Social Network", 2010,"David Fincher","Jeff Cronenweth", "2.39 : 1"),
            ("True Grit", 2010, "Joel Coen, Ethan Coen","Roger Deakins","2.35 : 1"),
            ("Hugo", 2011, "Martin Scorsese","Robert Richardson","1.85 : 1"),
            ("The Artist", 2011,"Michel Hazanavicius","Guillaume Schiffman","1.33 : 1"),
            ("The Girl with the Dragon Tattoo", 2011,"David Fincher","Jeff Cronenweth","2.39 : 1"),
            ("The Tree of Life", 2011,"Terrence Malick","Emmanuel Lubezki","1.85 : 1"),
            ("War Horse",2011,"Steven Spielberg","Janusz Kamiński","2.39 : 1"),
            ("Life of Pi",2012,"Ang Lee","Claudio Miranda","2.35 : 1"),
            ("Anna Karenina",2012,"Joe Wright","Seamus McGarvey","2.40 : 1"),
            ("Django Unchained",2012,"Quentin Tarantino","Robert Richardson","2.40 : 1"),
            ("Lincoln",2012,"Steven Spielberg","Janusz Kamiński","2.39 : 1"),
            ("Skyfall",2012,"Sam Mendes","Roger Deakins","1.90 : 1"),
            ("Gravity",2013,"Alfonso Cuarón","Emmanuel Lubezki","2.39 : 1"),
            ("The Grandmaster",2013,"Wong Kar-wai","Philippe Le Sourd","2.35 : 1"),
            ("Inside Llewyn Davis",2013,"Joel Coen, Ethan Coen","Bruno Delbonnel","1.85 : 1"),
            ("Nebraska",2013,"Alexander Payne","Phedon Papamichael","2.35 : 1"),
            ("Prisoners",2013,"Denis Villeneuve","Roger Deakins","1.85 : 1"),
            ("Birdman",2014,"Alejandro G. Iñárritu","Emmanuel Lubezki","1.85 : 1"),
            ("The Grand Budapest Hotel",2014,"Wes Anderson","Robert Yeoman","1.37 : 1, 1.85 : 1, 2.39 : 1"),
            ("Ida",2014,"Paweł Pawlikowski","Łukasz Żal, Ryszard Lenczewski","1.37 : 1"),
            ("Mr. Turner",2014,"Mike Leigh","Dick Pope","2.35 : 1"),
            ("Unbroken",2014,"Angelina Jolie","Roger Deakins","2.35 : 1"),
            ("The Revenant",2015,"Alejandro G. Iñárritu","Emmanuel Lubezki","2.39 : 1"),
            ("Carol",2015,"Todd Haynes","Edward Lachman","1.85 : 1"),
            ("The Hateful Eight",2015,"Quentin Tarantino","Robert Richardson","2.76 : 1"),
            ("Mad Max: Fury Road",2015,"George Miller","John Seale","2.39 : 1"),
            ("Sicario",2015,"Denis Villeneuve","Roger Deakins","2.39 : 1"),
            ("La La Land",2016,"Damien Chazelle","Linus Sandgren","2.55 : 1"),
            ("Arrival",2016,"Denis Villeneuve","Bradford Young","2.39 : 1"),
            ("Lion",2016,"Garth Davis","Greig Fraser","2.39 : 1"),
            ("Moonlight",2016,"Barry Jenkins","James Laxton","2.39 : 1"),
            ("Silence",2016,"Martin Scorsese","Rodrigo Prieto","2.39 : 1"),
            ("Blade Runner 2049",2017,"Denis Villeneuve","Roger Deakins","2.39 : 1"),
            ("Darkest Hour",2017,"Joe Wright","Bruno Delbonnel","1.85 : 1"),
            ("Dunkirk",2017,"Christopher Nolan","Hoyte van Hoytema","1.43 : 1, 1.78 : 1, 1.90 : 1, 2.20 : 1, 2.39 : 1"),
            ("Mudbound",2017,"Dee Rees","Rachel Morrison","2.39 : 1"),
            ("The Shape of Water",2017,"Guillermo del Toro","Dan Laustsen","1.85 : 1"),
            ("Roma",2018,"Alfonso Cuarón","Alfonso Cuarón","2.39 : 1"),
            ("Cold War",2018,"Paweł Pawlikowski","Łukasz Żal","1.37 : 1"),
            ("The Favourite",2018,"Yorgos Lanthimos","Robbie Ryan","1.85 : 1"),
            ("Never Look Away",2018,"Florian Henckel von Donnersmarck","Caleb Deschanel","1.85 : 1"),
            ("A Star Is Born",2018,"Bradley Cooper","Matthew Libatique","2.39 : 1"),
            ("1917",2019,"Sam Mendes","Roger Deakins","1.90 : 1, 2.39 : 1"),
            ("The Irishman",2019,"Martin Scorsese","Rodrigo Prieto","1.85 : 1"),
            ("Joker",2019,"Todd Phillips","Lawrence Sher","1.85 : 1"),
            ("The Lighthouse",2019,"Robert Eggers","Jarin Blaschke","1.19 : 1"),
            ("Once Upon a Time in Hollywood",2019,"Quentin Tarantino","Robert Richardson","2.39 : 1"),
            ("Judas and the Black Messiah",2020,"Shaka King","Sean Bobbitt","2.39 : 1"),
            ("Mank",2020,"David Fincher","Erik Messerschmidt","2.21 : 1"),
            ("Nomadland",2020,"Chloé Zhao","Joshua James Richards","2.39 : 1"),
            ("News of the World",2020,"Paul Greengrass","Dariusz Wolski","2.39 : 1"),
            ("The Trial of the Chicago 7",2020,"Aaron Sorkin","Phedon Papamichael","2.39 : 1")
            ]

    frames = getFrameDB()

    mycursor = db.cursor()

    #execute these two lines first, comment after excution, and uncomment the following code blocks in order
    mycursor.execute("DROP DATABASE mauruemc_color")
    mycursor.execute("CREATE DATABASE mauruemc_color")

    #just in case you create the table before
    #mycursor.execute("DROP TABLE Movies")

    #1
    #mycursor.execute("CREATE TABLE Movies(id int PRIMARY KEY AUTO_INCREMENT, movieName VARCHAR(50), year int, director VARCHAR(50), cinematographer VARCHAR(50), aspectRatio VARCHAR(50))")
    #mycursor.executemany("INSERT INTO Movies(movieName, year, director, cinematographer, aspectRatio) VALUES (%s, %s, %s, %s, %s)", movies)

    #mycursor.execute("DROP TABLE Frames")

    #2
    #mycursor.execute("CREATE TABLE Frames(movieID int REFERENCES Movies(id), frameID bigint, filePath VARCHAR(100), avg_R float, avg_G float, avg_B float, PRIMARY KEY(frameID, movieID))")
    #mycursor.executemany("INSERT INTO Frames(movieID, frameID, filePath, avg_R, avg_G, avg_B) VALUES (%s, %s, %s, %s, %s, %s)", frames)

    #mycursor.execute("DROP TABLE Palettes")
    
    #3
    #mycursor.execute("CREATE TABLE Palettes(frame bigint REFERENCES Frames(frameID), colorRank int, top10_R float, top10_G float, top10_B float, Kmeans_R float, Kmeans_G float, Kmeans_B float, random10_R float, random10_G float, random10_B float, PRIMARY KEY(frame, colorRank))")
    
    db.commit()

    #showing the datas
    #mycursor.execute("SELECT * FROM Movies WHERE aspectRatio LIKE '%2.39 : 1%'")
    #mycursor.execute("SELECT * FROM Frames")
    #result = mycursor.fetchall()
    #num_fields = len(mycursor.description)
    #field_names = [i[0] for i in mycursor.description]
    #print(field_names)

    #for value in result:
        #print(value)

    
except mysql.connector.Error as error:
    db.rollback()
    print("Failed to insert into MySQL table {}".format(error))

finally:
    if db.is_connected():
        mycursor.close()
        db.close()
        print("MySQL connection is closed")


