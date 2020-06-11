yadisk-direct https://yadi.sk/d/snd3nasL7n8y-Q >> ./models_url.txt
yadisk-direct https://yadi.sk/d/HyQxqZ88fB7Qtw >> ./data_url.txt
python3 load_data.py

unzip data.zip
unzip models.zip

rm data.zip
rm models.zip

mv ./best_models/ ./models/

mkdir ./logs