Reduction of Cost Sensitive MultiClass Classification to One to One Binary Classification

General description:

This code is a Python implementation of one vs one binary classifiers.
All the coding is done by the projectmates on their own. Dataset sources are properly given credits below.

One vs one binary classifiers: 
1. one vs one (OVO)
2. Cost sensitive one vs one (CSOVO)
3. Weighted all pairs (WAP)
4. Cost weighted neural network (CWNN)

Input:

Here 8 datasets are considered to compare efficiencies of various one vs one binary classifiers.
Datasets:
1. Title: Glass Identification Database
	Sources:
	(a) Creator: B. German
        	-- Central Research Establishment
           	Home Office Forensic Science Service
           	Aldermaston, Reading, Berkshire RG7 4PN
	(b) Donor: Vina Spiehler, Ph.D., DABFT
		--Diagnostic Products Corporation
		(213) 776-0180 (ext 3014)
	(c) Date: September, 1987
	
	Number of Instances: 214
	Number of Classes: 10

2. Title: Vehicle silhouettes
	Source:
		Drs.Pete Mowforth and Barry Shepherd
		Turing Institute
		George House
		36 North Hanover St.
		Glasgow
		G1 2AD
	Number of Instances: 946
	Number of Classes: 4

3. Title: Vowel Recognition (Deterding data)
	Source: 
		David Deterding  (data and non-connectionist analysis)
		Mahesan Niranjan (first connectionist analysis)
		Tony Robinson    (description, program, data, and results)
		To contact Tony Robinson by electronic mail, use address
		"tony@av-convex.ntt.jp" until 1 June 1989, and "ajr@dsl.eng.cam.ac.uk"
		thereafter
	Number of Instances: 990
	Number of Classes: 11

4. Title: Protein Localization Sites (Yeast dataset)
	Source:
		Kenta Nakai
             Institue of Molecular and Cellular Biology
	     Osaka, University
	     1-3 Yamada-oka, Suita 565 Japan
	     nakai@imcb.osaka-u.ac.jp
             http://www.imcb.osaka-u.ac.jp/nakai/psort.html
   	     Donor: Paul Horton (paulh@cs.berkeley.edu)
   	     Date:  September, 1996
	Number of Instances: 1484
	Number of Classes: 10

5. Title: Zoo database
	Source:
		Creator: Richard Forsyth
		Donor: Richard S. Forsyth 
	        8 Grosvenor Avenue
	        Mapperley Park
        	Nottingham NG3 5DX
                0602-621676
              	Date: 5/15/1990
	Number of Instances: 101
	Number of Classes: 7

6. Title: Pen-Based Recognition of Handwritten Digits Data Set
	Source:
		E. Alpaydin, Fevzi. Alimoglu
		Department of Computer Engineering
		Bogazici University, 80815 Istanbul Turkey 
	Number of Instances: 10992
	Number of Classes: 10

7. Title: Iris Plants Database
	Source:
		Creator: R.A. Fisher
		Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)
		Date: July, 1988
	Number of Instances: 150
	Number of Classes: 3

8. Title: Breast Cancer Wisconsin (Diagnostic) Data Set
	Source: 
		1. Dr. William H. Wolberg, General Surgery Dept.
		University of Wisconsin, Clinical Sciences Center
		Madison, WI 53792
		wolberg '@' eagle.surgery.wisc.edu

		2. W. Nick Street, Computer Sciences Dept.
		University of Wisconsin, 1210 West Dayton St., Madison, WI 53706
		street '@' cs.wisc.edu 608-262-6619

		3. Olvi L. Mangasarian, Computer Sciences Dept.
		University of Wisconsin, 1210 West Dayton St., Madison, WI 53706
		olvi '@' cs.wisc.edu

		Donor:

		Nick Street
	Number of Instances: 569
	Number of Classes: 2
	

Output: 

Here output are 4 metrices when an algorithm is choosen with a dataset
1. Simple accuracy
2. Cost sensitive accuracy
3. Average Mean cost
4. Average standard deviation

All these metrices are computed after randomly choosing 75% of the examples in each data set for training and leave the other 25% of the examples as the test set. The results reported are all averaged over 20 trials of different training/test splits, along with the standard error. In the report all these values are tabulated.

How to run:

1. Open terminal
2. run main.py by firing command : python main.py
3. Choose algorithm and dataset from Menu

Requirements:

1. python 2.7/3
2. pre-installed sklearn package
3. pre-installed numpy package

