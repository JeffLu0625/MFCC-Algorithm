#include<iostream>
#include<fstream>
#include<complex>
#include<vector>
#include <algorithm>
#include <functional> 
#include <time.h>
#include <map>
#include <numeric>
using namespace std;

double alpha = 0.97;
int numCepstra = 17;
double numFilters = 20;
double fs = 48000;
int winLength = 25;
int shiftLength = 10;
int lowFreq = 300;
int highFreq = 3700;
int origLength = 0;
int numTemplates = 0;
int numClass = 6;
double numFrames = 0;
double numCoef = 13;
double lifterParam = 22;
double frameDuration = round(1E-3*winLength*fs);
double frameShift = round(1E-3 * shiftLength * fs);
double n = floor(log(frameDuration) / log(2));
double numFFT = pow(2, pow(2, n) > frameDuration ? n : n + 1);
double k = floor(numFFT / 2) + 1;
const double PI = 4 * atan(1.0);
vector<vector<vector<double>>> Templates;
map<int, map<int, complex<double>> > omega;
clock_t t1, t2;

void ViewData(vector<double> arr) {
	for (size_t i = arr.size()-1; i >arr.size()-10; i--) {
		cout << arr[i] << endl;
	}
}

void SaveAsCSV(vector<vector<double>> arr, string fileName) {
	std::ofstream out(fileName);
	for (int i = 0; i < arr.size(); i++) {
		for (int k = 0; k < arr[0].size(); k++)
			out << arr[i][k] << ',';
		out << '\n';
	}
}

void ConvertCharToArray(const std::string array, vector<double>& OutputVertices) {
	std::istringstream ss(array);
	std::copy(
		std::istream_iterator <double>(ss),
		std::istream_iterator <double>(),
		back_inserter(OutputVertices)
	);
}

vector<vector<double>> matrixMultiply(vector<vector<double>> m1, vector<vector<double>> m2) {
	
	size_t initSize = m1.size();
	size_t lastSize = m2[0].size();
	size_t midSize = m1[0].size();
	vector<vector<double>> result(initSize, vector<double>(lastSize, 0));
	for (int k = 0; k < midSize; k++) {
		for (int i = 0; i < initSize; i++) {
			for (int j = 0; j < lastSize; j++) {
				if ((m1[i][k] == 0) || (m2[k][j] == 0))
					continue;
				result[i][j] += m1[i][k] * m2[k][j];
			}
		}
	}
	return result;
}

vector<vector<double>> matrixInverse(vector<vector<double>> m) {
	vector<vector<double>> result(m[0].size(), vector<double>(m.size(), 0));
	for (int i = 0; i < m[0] .size(); i++) {
		for (int j = 0; j < m.size(); j++) {
			result[i][j] = m[j][i];
		}
	}
	return result;
}

vector<double> SetAudioTo16Bits(vector<double> arr) {
	double maxVal = 0;
	double absArr = 0;
	size_t len = arr.size();
	for (int i = 0; i < len; i++) {
		absArr = abs(arr[i]);
		if (absArr > maxVal)
			maxVal = absArr;
	}
	if (maxVal <= 1)
		transform(arr.begin(), arr.end(), arr.begin(), bind(multiplies<double>(), placeholders::_1, pow(2, 15)));
	return arr;
}

vector<double> hz2mel(vector<double> hz) {
	size_t hzSize = hz.size();
	vector<double> mel(hzSize, 0);
	for (int i = 0; i < hzSize; i++) {
		mel[i] = 1127 * log(1 + hz[i] / 700);
	}
	return mel;
}

double hz2mel(double hz) {
	return 1127 * log(1 + hz / 700);
}

vector<double> mel2hz(vector<double> mel) {
	size_t melSize = mel.size();
	vector<double> hz(melSize, 0);
	for (int i = 0; i < melSize; i++) {
		hz[i] = 700 * exp(mel[i] / 1127) - 700;
	}
	return hz;
}

double mel2hz(double mel) {
	return 700 * exp(mel / 1127) - 700;
}

vector<double> Preemphasize(vector<double> speech, double alpha) {
	size_t speechLen = speech.size();
	vector<double> newSpeech(speechLen, 0);
	newSpeech[0] = speech[0];
	for (int i = 1; i < speechLen; i++) {
		newSpeech[i] = speech[i] - alpha * speech[(double)i - 1];
	}
	return newSpeech;
}

vector<vector<double>> vec2frame(vector<double> vec) {
	numFrames = floor((vec.size()-frameDuration)/frameShift+1);
	vector<vector<double>> frame(numFrames, vector<double>(frameDuration,0));
	// Framing and Hamming
	for (int i = 0; i < numFrames; i++) {
		for (int j = 0; j < frameDuration; j++) {
			frame[i][j] = vec[i* frameShift+j] * (0.54 - 0.46 * cos(2 * PI * j / (frameDuration - 1)));
		}
	}
	return frame;
}

void MappingOmega() {
	const complex<double> J(0, 1);      // Imaginary number 'j'
	for (int N = 2; N <= numFFT; N *= 2)
		for (int k = 0; k <= N / 2 - 1; k++)
			omega[N][k] = exp(-2 * PI * k / N * J);
}

vector<complex<double>> FFT(vector<complex<double>> frame) {
	const complex<double> J(0, 1);
	complex<double> tempY;
	int n = frame.size();
	vector<complex<double>> frame_even(n/2,0), frame_odd(n / 2, 0), y_odd, y;
	if (n == 1)
		return frame;
	for (int i = 0; i < n; i += 2) {
		frame_even[i / 2] = frame[i];
	}
	for (int i = 1; i < n ; i += 2)
		frame_odd[(i-1) / 2] = frame[i];
	y = FFT(frame_even);
	y_odd = FFT(frame_odd);
	y.insert(y.end(), y_odd.begin(), y_odd.end());
	for (int i = 0; i < n / 2-1; i++) {
		tempY = y[i];
		y[i] = tempY + omega[n][i] * y[(double)i+n/2];
		y[(double)i + n / 2] = tempY - omega[n][i] * y[(double)i+n/2];
	}
	return y;
}

vector<vector<double>> trifbank() {
	vector<vector<double>> H(numFilters, vector<double>(k, 0));
	double fl = hz2mel(lowFreq);
	double fh = hz2mel(highFreq);
	vector<double> hzVec(numFilters + 2, 0);
	for (int i = 0; i < numFilters + 2; i++) {
		hzVec[i] = fl + i* ((fh - fl) / (numFilters + 1));
	}
	vector<double> c = mel2hz(hzVec);
	vector<double> cw = hz2mel(c);
	
	vector<double> f(k,0);
	for (int i = 1; i < k; i++)
		f[i] = i*( (fs/2)/ (k - 1));
	for (size_t i = 0; i < numFilters; i++) {
		for (size_t j = 0; j < k; j++) {
			// Up-slope
			if ((f[j] >= c[i]) && (f[j] <= c[i + 1]))
				H[i][j] = (f[j] - c[i]) / (c[i + 1] - c[i]);
			// Down-slope
			if ((f[j] >= c[i+1]) && (f[j] <= c[i + 2]))
				H[i][j] = (c[i+2] - f[j]) / (c[i + 2] - c[i+1]);
		}
	}
	return H;
}

double DTW(vector<vector<double>> m1, vector<vector<double>> m2) {
	double error = 0;
	if ((m1.size() == m2.size()) && (m1[0].size() != m2[0].size())) {
		m1 = matrixInverse(m1);	
		m2 = matrixInverse(m2);
	}
	double m1Size = m1.size();
	double m2Size = m2.size();
	vector<vector<double>> m12(m1Size, vector<double>(m2Size, 0));
	vector<double> tempVec;
	vector<double> neighbors{0,0,0};
	// Compute largest error to be neighbor
	int LargestErr = 4096;
	// Compute error matrix
	double iCount = 0;
	for (int i = m1Size -1; i >= 0; i--) {
		for (int j = 0; j < m2Size; j++) {
			tempVec.clear();
			std::transform(m1[iCount].begin(), m1[iCount].end(), m2[j].begin(), std::back_inserter(tempVec),
				[](double element1, double element2) {return pow((element1 - element2), 2); });
			tempVec.shrink_to_fit();
			m12[i][j] = sqrt(accumulate(tempVec.begin(), tempVec.end(), 0.0));
			fill(neighbors.begin(), neighbors.end(), LargestErr);
			if ((iCount-1 >= 0) && (iCount - 1 < m1Size))
				neighbors[2] = m12[i + 1][j];
			if ((j - 1 >= 0) && (j - 1 < m2Size))
				neighbors[0] = m12[i][j - 1];
			if (((iCount-1 >= 0) && (iCount - 1 < m1Size)) && ((j - 1 >= 0) && (j - 1 < m2Size)))
				neighbors[1] = m12[i + 1][j - 1];
			if ((iCount != 0) || (j != 0))
				m12[i][j] += *min_element(neighbors.begin(), neighbors.end());
		}
		iCount += 1;
	}
	error = m12[0][m2Size - 1];
	return error;
}

void LoadTemplates(char* fileName) {
	ifstream filedata(fileName);
	string line;
	vector<double> numFrames, tempTemplate;
	vector<vector<double>> temp2DTemplate(numCoef, vector<double>());
	int lineCount = 1;
	if (filedata.is_open()) {
		while (getline(filedata, line)) {
			switch (lineCount) {
			case 1:
				numTemplates = stoi(line);
				break;
			case 2:
				ConvertCharToArray(line, numFrames);
				break;
			default:
				tempTemplate.clear();
				ConvertCharToArray(line, tempTemplate);
				for (int i = 0; i < numCoef; i++) {
					temp2DTemplate[i].assign(tempTemplate.begin() + i * numFrames[lineCount - 3],
						tempTemplate.begin() + (i + 1) * numFrames[lineCount - 3]);
				}
				Templates.push_back(temp2DTemplate);
				break;
			}
			lineCount += 1;
		}
	}
}

int main()
{
	t1 = clock();
	// Test
	//vector<vector<double>> m1{ {2,3,4},{1,5,7},{8,4,9}};
	//vector<vector<double>> m2{ {1,6},{8,2},{9,4} };
	//double error = DTW(m1,m2);

	// Load templates
	char filename[] = "MFCC_Templates.txt";
	LoadTemplates(filename);

	// Load data
	ifstream filedata("Data.txt");
	vector<double> sample;
	vector<vector<double>> frame;
	vector<complex<double>> a;
	string temp, all;
	double tempVal;
	if (filedata.is_open())
	{
		getline(filedata, all);
		istringstream line(all);
		while (getline(line, temp, ' ')) {
			tempVal = stod(temp);
			sample.push_back(tempVal);
			origLength++;
		}
	}

	// Preprocess
	sample = SetAudioTo16Bits(sample);

	// Preemphasize
	sample = Preemphasize(sample, alpha);
	
	// Framing and hamming
	frame = vec2frame(sample);
	
	// FFT
	MappingOmega();
	vector<vector<double>> Mag(numFFT, vector<double>(numFrames, 0));
	vector<complex<double>> tempFrame;
	for (int i = 0; i < numFrames; i++) {
		tempFrame.assign(frame[i].begin(), frame[i].end());
		tempFrame.resize(numFFT);
		tempFrame = FFT(tempFrame);
		for (int j = 0; j < numFFT; j++) {
			Mag[j][i] = abs(tempFrame[j]);
		}
	}
	//SaveAsCSV(Mag, "test2.csv");

	// Create filter bank
	vector<vector<double>> H = trifbank();

	// Apply filter bank
	vector<vector<double>> FBE = matrixMultiply(H, Mag);
	
	// Create DCT matrix
	vector <vector<double>> DCT(numCoef, vector<double>(numFilters, 0));
	for (size_t i = 0; i < numCoef; i++) {
		for (size_t j = 0; j < numFilters; j++) {
			DCT[i][j] = sqrt(2.0 / numFilters) * cos(i * (PI * ((j + 1) - 0.5) / numFilters));
		}
	}
	// Apply DCT
	for (int i = 0; i < numFilters; i++)
		for (int j = 0; j < numFrames; j++)
			FBE[i][j] = log(FBE[i][j]); // Take log for FBE
	vector<vector<double>> CC = matrixMultiply(DCT, FBE);

	// Create lifter computation
	/*vector<double> lifter(numCoef, 0);
	for (int i = 0; i < numCoef; i++)
		lifter[i] = 1 + 0.5 * lifterParam * sin(PI * i / lifterParam);*/
	
	// Apply lifter computation
	for (int i = 0; i < numCoef; i++) {
		for (int j = 0; j < numFrames; j++) {
			CC[i][j] *= 1 + 0.5 * lifterParam * sin(PI * i / lifterParam);
		}
	}
	
	// Cepstral mean and variance normalization
	vector<double> mean(numCoef, 0);
	vector<double> std(numCoef, 0);
	for (int i = 0; i < numCoef; i++) {
		mean[i] = accumulate(CC[i].begin(), CC[i].end(), 0.0) / numFrames;
		for (int j = 0; j < numFrames; j++) {
			std[i] += pow(CC[i][j] - mean[i], 2);
		}
		std[i] = sqrt(std[i] / (numFrames - 1));
	}
	for (int i = 0; i < numCoef; i++) {
		transform(CC[i].begin(), CC[i].end(), CC[i].begin(), bind(minus<double>(), placeholders::_1, mean[i]));
		transform(CC[i].begin(), CC[i].end(), CC[i].begin(), bind(divides<double>(), placeholders::_1, std[i]));
	}

	// DTW
	vector<double> ErrorVec(numClass, 0);
	for (int i = 0; i < numClass; i++) {
		for (int j = 0; j < numTemplates; j++) {
			ErrorVec[i] += DTW(Templates[i*numTemplates+j], CC);
		}
	}
	

	//SaveAsCSV(CC, "CC.csv");
	t2 = clock();
	printf("%lf\n", (t2 - t1) / (double)(CLOCKS_PER_SEC));
	system("pause");
	return 0;
}