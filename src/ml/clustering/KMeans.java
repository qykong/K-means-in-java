package ml.clustering;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Scanner;

import nlp.nicta.filters.SnowballStemmer;
import text.UnigramBuilder;
import util.DocUtils;
import util.FileFinder;

public class KMeans {
	
	public static final String SPLIT_TOKENS = "[!\"#$%&'()*+,./:;<=>?\\[\\]^`{|}~\\s]"; // missing: [_-@]

	public final static String DATA_SRC = "/home/quyu_kong/Downloads/LabML/data/two_newsgroups";
	
	public final static boolean REMOVE_STOPWORDS = true;
	public final static int     NUM_TOP_WORDS    = 1000;
	public final static double  EPSILON          = 0.0000001d;
	
	public final ArrayList<Map<String, Double>> fileWordFrequency = new ArrayList<>();
	public final ArrayList<String> fileNames = new ArrayList<>();
	public final ArrayList<ArrayList<Integer>> clusters = new ArrayList<>();
	public final ArrayList<Map<String, Double>> centroids = new ArrayList<>();
	
	public KMeans(String directory, HashMap<String,Integer> feature2index, int k) {
		//create clusters and will store the index of file into clusters
		for (int i = 0; i < k; i++) {
			clusters.add(new ArrayList<>());
			centroids.add(new HashMap<String, Double>());
		}
		
		ArrayList<File> files = FileFinder.GetAllFiles(directory, "", true);
		SnowballStemmer sbs = new SnowballStemmer();
		//build vector of every file
		for (File file: files) {
			String file_content = DocUtils.ReadFile(file);
			Map<String, Double> word_counts = new HashMap<String,Double>();
			
			String tokens[] = file_content.split(SPLIT_TOKENS);
			//ArrayList<String> tokens = ST.extractTokens(sent, true);
			for (String token : tokens) {
				token = token.trim().toLowerCase();
				token = sbs.stem(token);
				if (!feature2index.containsKey(token) || token.length() == 0)
					continue;
				if (word_counts.containsKey(token))
					word_counts.put(token, word_counts.get(token) + 1d);
				else
					word_counts.put(token, 1d);
			}
			fileWordFrequency.add(word_counts);
			fileNames.add(file.getName() + " " + file.getPath());
			if (word_counts.size() == 0) {
				System.out.println("This is empty: " + file.getName());
			}
		}
		//Count inverse document frequency
		for (Map.Entry<String, Integer> me : feature2index.entrySet()) {
			int df = 0;
			String word = me.getKey();
			for (Map<String, Double> m : fileWordFrequency) {
				if (m.containsKey(word)) df++;
			}
			double idf = Math.log10((double)fileWordFrequency.size() / (double)df);
			for (Map<String, Double> m : fileWordFrequency) {
				if (m.containsKey(word)) m.put(word, m.get(word) * idf);
			}
		}
		
		randomInitClusters();
		
		boolean running = true;
		//k-means
		while (running) {
			running = false;
			int whichFile = 0;
			for (int a = 0; a < k; a++) clusters.get(a).clear();
			//find clusters for every doc
			for (Map<String, Double> m: fileWordFrequency) {
				double tmp = Double.POSITIVE_INFINITY;
				int whichCentroid = 0;
				int i = 0;
				
				//find which cluster this file belongs to
				for (Map<String, Double> centroid: centroids) {
					double sim = cosineSim(m, centroid);
					if (tmp - sim > EPSILON) {
						whichCentroid = i;
						tmp = sim;
					}
					i++;
				}
				clusters.get(whichCentroid).add(whichFile);
				whichFile++;
			}
			
			//recompute centroids
			boolean details = false;
			for (int j = 0; j < k; j++) {
				Map<String, Double> newCentroid = new HashMap<>();
				for (Integer doc: clusters.get(j)) {
					for (Map.Entry<String, Double> me: fileWordFrequency.get(doc).entrySet()) {
						if (newCentroid.containsKey(me.getKey())) 
							newCentroid.put(me.getKey(), me.getValue() + newCentroid.get(me.getKey()));
						else
							newCentroid.put(me.getKey(), me.getValue());
					}
				}
				for (Map.Entry<String, Double> m : newCentroid.entrySet()) {
					newCentroid.put(m.getKey(), (m.getValue()/clusters.get(j).size()));
				}
				
				if (!running && centroids.get(j).size() != newCentroid.size()) {
					running = true;
					details = true;
				}
				//compare new controids to old controids
				if (!running && details) {
					for (Map.Entry<String, Double> m : newCentroid.entrySet()) {
					running = (!centroids.get(j).containsKey(m.getKey())) || 
								(Math.abs(centroids.get(j).get(m.getKey()) - m.getValue()) > EPSILON);
					}
				}
				centroids.set(j, newCentroid);
			}
			
		}
		for (int o=0; o < k; o ++)
			System.out.println("There are "+clusters.get(o).size()+" files in the "+(o+1)+" cluster.");
	}
	
	public void printTop(int num) {
		for (int k = 0; k < centroids.size(); k++) {
			Double sims[] = new Double[num];
			int simd[] = new int[num];
			for (int j = 0; j < num; j++) sims[j] = Double.POSITIVE_INFINITY;
			System.out.println("This is top "+num+" documents in cluster "+(k+1)+":");
			
			//get top documents
			for (Integer doc : clusters.get(k)) {
				Double sim = cosineSim(fileWordFrequency.get(doc), centroids.get(k));
				for (int j = 0; j < num; j++) {
					if (sims[j] - sim > EPSILON) {
						for (int i = num-1; i > j; i--) {
							sims[i] = sims[i-1];
							simd[i] = simd[i-1];
						}
						sims[j] = sim;
						simd[j] = doc;
						break;
					}
				}
			}
			
			for (int j = 0; j < num; j++) {
				System.out.println((j+1)+": "+fileNames.get(simd[j]));
			}
		}
	}
	
//	private double euclidean(Map<String, Double> m1, Map<String, Double> m2) {
//		Map<String, Double> m = new HashMap<String,Double>(m1);
//		Double euc = 0d;
//		for (Map.Entry<String, Double> k : m2.entrySet()) {
//			if (m.containsKey(k.getKey())) {
//				m.put(k.getKey(), k.getValue() - m.get(k.getKey()));
//			} else {
//				m.put(k.getKey(), k.getValue());
//			}
//		}
//		for (Map.Entry<String, Double> k : m.entrySet()) {
//			euc += Math.pow(k.getValue(), 2);
//		}
//		return Math.sqrt(euc);
//	}
	
	//compute cosine similarity
	private double cosineSim(Map<String, Double> m1, Map<String, Double> m2) {
		//let m1 have smaller key set
		if (m1.size() > m2.size()) {
			Map<String, Double> swap = m1;
			m1 = m2;
			m2 = swap;
		}
		double innerProduct = 0d;
		
		for (Map.Entry<String, Double> me: m1.entrySet()) {
			if (m2.containsKey(me.getKey())) {
				innerProduct += me.getValue() * m2.get(me.getKey());
			}
		}

		return 1d - (innerProduct / (magnitude(m1) * magnitude(m2)));
	}
	
	private double magnitude(Map<String, Double> m) {
		double add = 0d;
		
		for (Map.Entry<String, Double> me: m.entrySet()) {
			add += Math.pow(me.getValue(), 2);
		}
		
		return Math.sqrt(add);
	}
	
	//select random seeds and initial the controids
	private void randomInitClusters() {
		int numOfDocs = fileWordFrequency.size();
		Random rd = new Random();
		List<Integer> randomDocs = new ArrayList<>();
		
		while (randomDocs.size() < clusters.size()) {
			int j; 
			do {
				j = rd.nextInt(numOfDocs);
			} while (randomDocs.contains(j));
			randomDocs.add(j);
		}
		
		for (int i = 0; i < clusters.size(); i++) centroids.get(i).putAll(fileWordFrequency.get(randomDocs.get(i)));
	}
	
	public static void main(String[] args) {
		UnigramBuilder ub = new UnigramBuilder(
				DATA_SRC, 
				NUM_TOP_WORDS,
				REMOVE_STOPWORDS);
		int k;
		Scanner sc = new Scanner(System.in);
		System.out.println("Please enter the number of k:");
		k = sc.nextInt();
		sc.close();
		KMeans km = new KMeans(DATA_SRC, ub._topWord2Index, k);
		km.printTop(5);
	}

}
