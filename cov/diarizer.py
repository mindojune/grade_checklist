import json
import glob
from collections import Counter

corpus_root = "./google/"
for file_path in glob.glob('{}/*.json'.format(corpus_root)):
    #print(file_path)
    #continue

    with open(file_path) as fh:
        temp = json.load(fh)
    #print(temp["results"])
    transcripts = []
    speakertags = []
    for item in temp["results"]:
        #print(item)
        #exit()
        if "alternatives" in item:
            if "transcript" in item["alternatives"][0]:
                text = item["alternatives"][0]["transcript"]
                #print(text)
                transcripts.append(text)
            else:
                
                words = item["alternatives"][0]["words"]
                for word in words:
                    #print(word)
                    tag = word["speakerTag"]
                    speakertags.append(tag)
                    #print(tag)

    #print(len(transcripts))
    #print(len(speakertags))
    
    tindex = 0
    sindex = 0

    tags = []
    for utt in transcripts:
        length = len(utt.split())
        stags = speakertags[sindex:sindex+length]
        #print(utt)
        #print(stags)
        tag = Counter(stags).most_common(1)[0][0]     
        #print(tag)
        #exit()
        tags.append(tag)
        sindex += length
    assert(sindex == len(speakertags))
    assert(len(tags) == len(transcripts))

    with open(file_path[:-4]+"txt", "w") as fh:
        for utt,tag in zip(transcripts, tags):

            fh.write(str(tag)+": "+utt)
            fh.write("\n")

    




