import pymongo

client = pymongo.MongoClient("mongodb+srv://gogorun235:nhathuy@rag-publicsector.amgor2s.mongodb.net/")

db = client["rag-db"]

# Chọn collection "documents"
collection = db["documents"]

# Bây giờ bạn có thể thực hiện các thao tác với collection "documents", ví dụ:
# - Thêm dữ liệu
# - Truy vấn dữ liệu
# - Cập nhật dữ liệu
# - Xóa dữ liệu
# Ví dụ:
document = {"title": "Sample Document", "content": "This is a sample document"}
collection.insert_one(document)
