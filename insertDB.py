from pymongo import MongoClient


def insertEmployee(id,name,gender,email):
    client = MongoClient('mongodb://localhost:27017/')
    database = client['face_db']
    collection = database['data']
    
    collection.insert_one({
        '_id':3,
        'empId':id,
        'empName':name,
        'empGender':gender,
        'empEmail':email
    })
    
    client.close()
    
# insertEmployee(12108,'vicky','Male','vicky@gmail.com')

def findEmployee(empName):
    client = MongoClient('mongodb://localhost:27017/')
    database = client['face_db']
    collection = database['data']
    
    employeeData = collection.find_one({'empName':empName})
    
    client.close()
    
    return employeeData
     