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
int fs = 48000;
int winLength = 25;
int shiftLength = 10;
int lowFreq = 300;
int highFreq = 3700;
int origLength = 0;
int numFrames = 0;
double numCoef = 13;
double lifterParam = 22;
double frameDuration = round(1E-3*winLength*fs);
double frameShift = round(1E-3 * shiftLength * fs);
double n = floor(log(frameDuration) / log(2));
double numFFT = pow(2, pow(2, n) > frameDuration ? n : n + 1);
double k = floor(numFFT / 2) + 1;
const double PI = 4 * atan(1.0);
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

vector<vector<double>> matrixMultiply(vector<vector<double>> m1, vector<vector<double>> m2) {
	
	int initSize = m1.size();
	int lastSize = m2[0].size();
	int midSize = m1[0].size();
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
	int len = arr.size();
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
	int hzSize = hz.size();
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
	int melSize = mel.size();
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
	int speechLen = speech.size();
	vector<double> newSpeech(speechLen, 0);
	newSpeech[0] = speech[0];
	for (int i = 1; i < speechLen; i++) {
		newSpeech[i] = speech[i] - alpha * speech[i - 1];
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
	size_t n = frame.size();
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
		y[i] = tempY + omega[n][i] * y[i+n/2];
		y[i + n / 2] = tempY - omega[n][i] * y[i+n/2];
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
		f[i] = i*( (double(fs)/2)/ ((double)k - 1));
	for (int i = 0; i < numFilters; i++) {
		for (int j = 0; j < k; j++) {
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
	vector<vector<double>> m12(m1.size(), vector<double>(m2.size(), 0));
	vector<double> tempVec;
	vector<double> neighbors;
	// Compute error matrix
	int iCount = 0;
	for (int i = m1.size()-1; i > 0; i--) {
		for (int j = 0; j < m2.size(); j++) {
			tempVec.clear();
			std::transform(m1[iCount].begin(), m1[iCount].end(), m2[j].begin(), std::back_inserter(tempVec),//
				[](double element1, double element2) {return pow((element1 - element2), 2); });
			tempVec.shrink_to_fit();
			m12[i][j] = sqrt(std::accumulate(tempVec.begin(), tempVec.end(), 0));
			// Add neighbors' condition
		}
		iCount += 1;
	}
	return error;
}
int main()
{
	//t1 = clock();
	// Test
	vector<vector<double>> m1{ {2,3,4},{1,5,7},{8,4,9}};
	vector<vector<double>> m2{ {1,6},{8,2},{9,4} };
	double error = DTW(m1,m2);

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
	for (int i = 0; i < numCoef; i++) {
		for (int j = 0; j < numFilters; j++) {
			DCT[i][j] = sqrt(2.0 / numFilters) * cos((double)i * (PI * (((double)j + 1) - 0.5) / numFilters));
		}
	}
	// Apply DCT
	for (int i = 0; i < numFilters; i++)
		for (int j = 0; j < numFrames; j++)
			FBE[i][j] = log(FBE[i][j]); // Take log for FBE
	vector<vector<double>> CC = matrixMultiply(DCT, FBE);

	// Create lifter computation
	vector<double> lifter(numCoef, 0);
	for (int i = 0; i < numCoef; i++)
		lifter[i] = 1 + 0.5 * lifterParam * sin(PI * i / lifterParam);
	
	// Apply lifter
	for (int i = 0; i < numCoef; i++)
		for (int j = 0; j < numFrames; j++)
			CC[i][j] *= lifter[i];


	//SaveAsCSV(CC, "CC.csv");
	//t2 = clock();
	//printf("%lf\n", (t2 - t1) / (double)(CLOCKS_PER_SEC));
	system("pause");
	return 0;
}