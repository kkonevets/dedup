#!/usr/bin/env bash

#docker exec -u 0 -it solonom /bin/bash

docker stop solonom || true && docker rm solonom || true

schema="http://localhost:8983/solr/nom_core/schema"

cd ../..
docker run --name solonom -d -p 8983:8983 -t -v $PWD/data/solr:/opt/solr/mydata solr
cd dedup/solr/
docker exec -it --user=solr solonom bin/solr create_core -c nom_core

docker cp stopwords_ru.txt solonom:/opt/solr/server/solr/nom_core/conf/lang/stopwords_ru.txt

      # "tokenizer": {
      #   "class": "solr.StandardTokenizerFactory"
      #   "class": "solr.WhitespaceTokenizerFactory"
      # },

      # {
      #   "class": "solr.PatternReplaceFilterFactory",
      #   "pattern": "\\d+",
      #   "replacement": ""
      # },

curl -X POST -H 'Content-type:application/json' --data-binary '{
  "add-field-type": {
    "name": "text_general_ru",
    "class": "solr.TextField",
    "positionIncrementGap": "100",
    "analyzer": {
      "tokenizer": {
        "class": "solr.WhitespaceTokenizerFactory"
      },
      "filters": [
        {
          "class": "solr.LowerCaseFilterFactory"
        },
        {
          "class": "solr.StopFilterFactory",
          "ignoreCase": true,
          "words": "lang/stopwords_ru.txt",
          "format": "snowball"
        },
        {
          "class": "solr.LengthFilterFactory",
          "min": "2",
          "max": "100"
        },
        {
          "class": "solr.SnowballPorterFilterFactory",
          "language": "Russian"
        }
      ]
    }
  }
}' $schema


# curl -X POST -H 'Content-type:application/json' --data-binary '{
#   "add-field-type": {
#     "name": "text_general_ru",
#     "class": "solr.TextField",
#     "positionIncrementGap": "100",
#     "analyzer": {
#       "tokenizer": {
#         "class": "solr.WhitespaceTokenizerFactory"
#       },
#       "filters": [
#         {
#           "class": "solr.LowerCaseFilterFactory"
#         },
#         {
#           "class": "solr.StopFilterFactory",
#           "ignoreCase": true,
#           "words": "lang/stopwords_ru.txt",
#           "format": "snowball"
#         },
#         {
#           "class": "solr.EdgeNGramTokenizerFactory",
#           "minGramSize": "3",
#           "maxGramSize": "8"
#         }
#       ]
#     }
#   }
# }' $schema


# "add-field": {"name":"my_text_ru", "type":"text_general_ru", "indexed":true, "multiValued":true, "stored":false},

curl -X POST -H 'Content-type:application/json' --data-binary \
'{
"add-field": {"name":"name", "type":"text_general_ru", "indexed":true, "multiValued":false, "stored":true},
"add-field": {"name":"synonyms", "type":"text_general_ru", "indexed":true, "multiValued":false, "stored":true},

"add-field": {"name":"manufacturer", "type":"text_general_ru", "multiValued":false, "stored":true},
"add-field": {"name":"unitName", "type":"text_general_ru", "multiValued":false, "stored":true},

"add-field": {"name":"categoryId", "type":"plong", "multiValued":false, "indexed":true, "stored":true},
"add-field": {"name":"brandId", "type":"plong", "multiValued":false, "indexed":true, "stored":true},
"add-field": {"name":"manufacturerId", "type":"plong", "multiValued":false, "indexed":true, "stored":true},
"add-field": {"name":"barcodes", "type":"text_general_ru", "multiValued":true, "indexed":true, "stored":true},
"add-field": {"name":"manufacturerCode", "type":"text_general_ru", "multiValued":false, "stored":true}
}' $schema

# curl -X POST -H 'Content-type:application/json' --data-binary \
# '{
# "add-copy-field" : {"source":"name","dest":"my_text_ru"},
# "add-copy-field" : {"source":"brand","dest":"my_text_ru"}
# }' $schema


docker exec -it --user=solr solonom bin/post -c nom_core  /opt/solr/mydata/release_data.json
