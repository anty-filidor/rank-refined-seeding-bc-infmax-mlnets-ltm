# ltm-seeding-mln
Repository for experiments on seeding methods for linear threshold model on multilayer networks


# repo with ICM experiments
https://github.com/pbrodka/SQ4MLN

# DATA
AUCS | http://multilayer.it.uu.se/datasets.html; other sources https://networks.skewed.de/net/cs_department, https://github.com/pbrodka/SQ4MLN/tree/master/FullNet
Ckm Physicians Innovation | https://figshare.com/articles/dataset/CKM-Physicians-Innovation_Multiplex_Social/21545784
EU Transportation | http://complex.unizar.es/~atnmultiplex/, https://networks.skewed.de/net/eu_airlines
Lazega Law Firm | https://networks.skewed.de/net/law_firm, http://elazega.fr/?page_id=609, https://search.r-project.org/CRAN/refmans/sand/html/lazega.html
http://konect.cc/networks/


https://networks.skewed.de/


# Config runtime

git submodule update --init --recursive

conda create --name ltm-seeding-mln python=3.10
conda activate ltm-seeding-mln
pip install -r submodules/network-diffusion/requirements/production.txt


### Unix
ln -s network_diffusion submodules/network-diffusion/network_diffusion

### Windows
mklink /J .\network_diffusion .\submodules\network-diffusion\network_diffusion