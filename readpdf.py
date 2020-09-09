import fitz
filename = 'test.pdf'
doc = fitz.open(filename)
print("Number of pages in document: {}".format(doc.pageCount))
print("Document metadata:{}".format(doc.metadata))
first_page = doc.loadPage(0)
print("Content of first page:")
print(first_page.getText())