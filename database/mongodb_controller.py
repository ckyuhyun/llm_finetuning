from typing import Optional
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import utility

DEBUG_MODE = True

class MongodbController:
    def __init__(self):
        self.db = None
        self.collection = None
        self._data_id = None
        self._db_name = "my_db"
        self.collection_name = 'collection_1'
        # Replace the placeholder with your Atlas connection string
        self.mongo_db_url ="mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.3.3"

        self.connect_mongodb()
        self.create_collection()


    def connect_mongodb(self):
        # Set the Stable API version when creating a new client
        client =  MongoClient(self.mongo_db_url,
                              server_api=ServerApi('1'),
                              directConnection=True)

        self.db = client[self._db_name]

    def create_collection(self):
        try:
            if self.collection is None or not self.collection_name in self.collection.find():
                self.collection = self.db[self.collection_name]

            if DEBUG_MODE:
                print('########## Existing Data ###########')
                for x in self.collection.find():
                    print(x)

        except Exception as e:
            utility.except_error_print(e)


    async def insert_data(self, query_id:int, data:dict):
        self._attach_query_id_to_data(query_id, data)

        # if true, which means there is one or more data in database for the id
        id_query = self._get_query_only_for_id(query_id)
        document_count = self._get_document_count(id_query)
        if document_count == 0:
            try:
                self.collection.insert_one(self._data_id)
            except Exception as e:
                utility.except_error_print(e)
        else:
            # This is only considering replacing one to one
            if document_count == 1:
                await self.replace_data_one(id_query, data)
            else:
                utility.except_error_print('It should not have data more than one')








    async def replace_data_one(self,
                               filter_query:dict,
                               new_query:dict):

        if self._get_document_count(filter_query) == 1:
            old_document = self.collection.find_one(filter_query)
            # id for the filtered query
            _id = old_document.get('_id')
            new_query.update({"_id": _id})

            self.collection.replace_one(old_document, new_query)
        else:
            print(f'{filter_query} has more data in the database')


    async def get_data(self,
                       query_id:int,
                       query:dict) \
            -> list:
        self._attach_query_id_to_data(query_id, query)

        document_list=[]
        for document in self.collection.find(self._data_id):
            document_list.append(document)
            return document_list



    async def delete_data_many(self,
                               query: dict,
                               query_id:Optional[int]=None):
        try:
            if query_id is None:
                self._data_id = dict(query)
            else:
                self._attach_query_id_to_data(query_id, query)

            if self.collection.count_documents(self._data_id) == 0:
                print('Collection does not have this data\n' + f'{query}\n')
            else:
                self.collection.delete_many(self._data_id)
                print('data is deleted')
        except Exception as e:
            utility.except_error_print(e)

        if DEBUG_MODE:
            print('############## DEBUG ###########')
            for d in self.collection.find():
                print(d)
            print('#########################')

    def _get_document_count(self,data:dict) \
            -> int:
        return self.collection.count_documents(data)

    def _attach_query_id_to_data(self, query_id: int, query:dict):
        _query = dict(query)
        _query.update(self._get_query_only_for_id(query_id))
        self._data_id = _query



    def _get_query_only_for_id(self, query_id) \
            -> dict:
        return {'_id': query_id}


    # async def ping_server(client):
    #
    #     # Send a ping to confirm a successful connection
    #     try:
    #         client.admin.command('ping')
    #         print("Pinged your deployment. You successfully connected to MongoDB!")
    #     except Exception as e:
    #         print(e)








