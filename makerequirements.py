x=[
"absl-py==0.9.0",
"gym==0.17.1",
"pygame==1.9.6",
"matplotlib==3.0.3",
"seaborn==0.9.1",
"pandas==0.25.3",
"scipy==1.4.1",
"transforms3d==0.3.1",
"tqdm==4.42.1",
"wget==3.2",
"networkx==2.4",
"imageio==2.8.0",
"tabulate==0.8.7",
"scikit-image==0.15.0",
"dm-tree==0.1.5",
    ]
with open("requirements.txt",	"w") as file:
    for line in x:
        file.write(line+"\n")
