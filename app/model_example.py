import boto3

#create a Textract Client
textract = boto3.client('textract', region_name='us-east-1')
# Document
documentName = "image.png"
with open(documentName, 'rb') as document:
    imageBytes = bytearray(document.read())

    # Call Textract
    response = textract.analyze_document(
        Document={'Bytes': imageBytes},
        FeatureTypes=["FORMS", "QUERIES"],
        QueriesConfig={
            "Queries": [{
                "Text": "What is the full name?",
                "Alias": "name"
            }]
        })

    print(response)