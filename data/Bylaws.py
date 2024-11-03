import geopandas as gpd
import json
import requests
import textract
import spacy
import pandas as pd

nlp_model = spacy.load('en_core_web_sm')


class Bylaws:
    def __init__(self):
        self.gdf = gpd.read_file(
            'https://opendata.vancouver.ca/explore/dataset/zoning-districts-and-labels/download/?format=geojson&timezone=America/Los_Angeles&lang=en',
            driver='GeoJSON'
        )
        self.bylaws_urls = {
            'commercial': [
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-c-1.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-c-2.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-c-2b.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-c-2c.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-c-2c1.pdf",
                "http://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-c-3a.pdf",
                "http://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-c-5-5a-6.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-c-7-8.pdf",
            ],
            'historic': [
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-ha-1-1a.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-ha-2.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-ha-3.pdf",
            ],
            'industrial': [
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-i-1.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-i-1a.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-i-1b.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-i-2.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-i-3.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-i-4.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-ic-1-2.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-ic-3.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-m-1.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-m-1a.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-m-1b.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-m-2.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-mc-1-2.pdf",
            ],
            'residential': [
                # Single-Family
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rs-1.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rs-1a.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rs-1b.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rs-2.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rs-3-3a.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rs-5.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rs-6.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rs-7.pdf",

                # Duplexes
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rt-1.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rt-2.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rt-3.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rt-4-all-districts.pdf",
                "http://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rt-5.pdf",
                "http://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rt-5.pdf",
                "http://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rt-6.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rt-7.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rt-8.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rt-9.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rt-10.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rt-11.pdf",

                # Multi-Family
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rm-1.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rm-2.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rm-3.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rm-3a.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rm-4.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rm-5-all-districts.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rm-6.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rm-7-7an.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rm-8-all-districts.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rm-9-all-districts.pdf",
                "http://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rm-10.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rm-11.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-rm-12n.pdf",
                "https://bylaws.vancouver.ca/zoning/zoning-by-law-district-schedule-fm-1.pdf"
                ],
            }
        return

    def boundaries(self):
        # Access Metro Vancouver Regional District web page to load the GeoDataFrame
        zoning_gdf = self.gdf

        # Generate unique id for each feature
        zoning_gdf['id'] = (zoning_gdf.index+1).astype(str)

        # Print a list of ids as a json and save to 'groups' folder
        with open('../other/groups/zoning.json', 'w') as file:
            file.write(str(list(zoning_gdf['id'])))

        # Export the GeoDataFrame as geojson.json to the 'geojson' folder
        zoning_gdf.to_file('../other/data/geojson/zoning.geojson.json', driver='GeoJSON')

        # Calculate area of zone
        zoning_gdf['area'] = zoning_gdf.to_crs(26910).area

        # Export selected metrics to be visualized
        data = {"map": {}}
        for i, metric in enumerate(['area']):
            for j in zoning_gdf['id']:
                j = int(j) - 1
                data["map"][j] = {"y_2020": round(zoning_gdf.loc[j, metric],2)}

            with open(f'../other/data/metric/m{i}.json', 'w') as file:
                json.dump(data, file)

        return zoning_gdf

    def bylaws(self):
        urls = self.bylaws_urls

        # Extract 'burdens'
        def extract_num(sentences, burdens, suffix=''):
            run = False
            splitter = 1
            num_list = []
            for sentence in sentences:
                for burden in burdens:
                    if type(burden).__name__ == 'list':
                        c = []
                        for b in burden:
                            if b in sentence:
                               c.append(b)
                        if len(c) == len(burden):
                            run = True
                            burden = burden[len(burden)-1]
                        else: run = False
                    else:
                        if burden in sentence: run = True
                        else: run = False
                    if run:
                        tokenize = sentence.split(burden)[1]
                        tokens = nlp_model(tokenize)
                        if suffix == '': num_list.append([token for n, token in enumerate(tokens) if (token.pos_ == "NUM")])
                        else:
                            num_list.append([token for n, token in enumerate(tokens) if (token.pos_ == "NUM") and (token != tokens[len(tokens)-1]) and (suffix in str(tokens[n+1]))])

            return num_list

        metrics = pd.DataFrame()
        for use, url_list in urls.items():
            for url in url_list:

                # Download the file
                resp = requests.get(url)
                split = url.split('/')
                filename = split[len(split)-1]

                # Write pdf
                with open(f"bylaws/{filename}", 'wb') as f:
                    f.write(resp.content)

                # Extract text from pdf
                raw_text = textract.process(f"bylaws/{filename}").decode()

                # Split into sentences
                kg = KnowledgeGraph()
                sentences = [s.replace('\n', ' ') for s in kg.getSentences(raw_text)]

                max_height = extract_num(sentences, suffix='m', burdens=[
                    "building height must not exceed ",
                    "Height shall not exceed",
                    "maximum height of a building shall be ",
                    "A building shall not exceed",
                    "The height of a building shall not exceed",
                    "maximum height of a building shall not exceed",
                    "maximum permitted height of a building is",
                    "two-family dwelling with secondary suite shall not exceed",
                    "in RS-3, exceed ",
                    "A principal building shall not exceed",
                    "A building must not exceed",
                    "A Multiple Dwelling of four or more dwelling units must not exceed"
                ])

                fsr = extract_num(sentences, burdens=[
                    "The maximum floor space ratio shall not exceed",
                    "the floor space ratio shall not exceed",
                    ["The floor space ratio", "shall not exceed "],
                    "for all combined uses, up to",
                    "Maximum floor space ratio shall not exceed",
                    "The floor space ratio must not exceed",
                    "Floor space ratio must not exceed",
                    "The maximum floor space ratio shall be",
                    "floor space ratio must not exceed"
                ])

                i = len(metrics)

                # Land use
                metrics.at[i, 'land_use'] = use

                # Zone name
                zone_name = sentences[0].split(' ')[0]
                try: zone_name = zone_name.split(',')[0]
                except: pass
                metrics.at[i, 'zone_name'] = zone_name
                metrics.at[i, 'file'] = filename

                # Maximum height
                if len(max_height) > 0:
                    if len(max_height[0]) > 0: metrics.at[i, 'max_height'] = float(max_height[0][0].string.strip())

                # Maximum fsr
                if len(fsr) > 0:
                    for j in fsr[0]:
                        try:
                            j = float(j.string.strip())
                            if (j > 0) and (j < 30): metrics.at[i,'fsr'] = j
                        except: pass

            # Change zone names type to string
            metrics['zone_name'] = metrics['zone_name'].astype(str)

            # # Replace unfound values
            # metrics['max_height'] = metrics['max_height'].replace(np.nan, 4)

            metrics.loc[metrics['zone_name'] == 'HA-1', 'max_height'] = 15.2

        print("Metrics calculated")
        return metrics

