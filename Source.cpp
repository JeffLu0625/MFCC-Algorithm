#include<iostream>
#include<fstream>
#include<complex>
#include<vector>
#include <algorithm>
#include <functional> 
#include <time.h>
#include <map>
using namespace std;

double alpha = 0.97;
int numCepstra = 17;
int numFilters = 20;
int fs = 48000;
int winLength = 25;
int shiftLength = 10;
int lowFreq = 300;
int highFreq = 3700;
int origLength = 0;
int numFrames = 0;
double frameDuration = round(1E-3*winLength*fs);
double frameShift = round(1E-3 * shiftLength * fs);
int n = floor(log(frameDuration) / log(2));
int numFFT = pow(2, pow(2, n) > frameDuration ? n : n + 1);
int k = floor(numFFT / 2) + 1;
const double PI = 4 * atan(1.0);
map<int, map<int, complex<double>> > omega;

void ViewData(vector<double> arr) {
	for (int i = arr.size()-1; i >arr.size()-10; i--) {
		cout << arr[i] << endl;
	}
}

void SaveAsCSV(vector<double> arr, string fileName) {
	std::ofstream out(fileName);
	for (int k = 0; k < numFFT; k++) {
		out << arr[k] << '\n';
	}
}

vector<vector<double>> matrixMultiply(vector<vector<double>> m1, vector<vector<double>> m2) {
	//clock_t t1, t2;
	//t1 = clock();
	int initSize = m1.size();
	int midSize = m2[0].size();
	int lastSize = m2.size();
	vector<vector<double>> result(initSize, vector<double>(lastSize, 0));
	for (int k = 0; k < lastSize; k++) {
		for (int i = 0; i < initSize; i++) {
			for (int j = 0; j < midSize; j++) {
				if ((m1[i][k] == 0) || (m2[k][j] == 0))
					continue;
				result[i][j] += m1[i][k] * m2[k][j];
			}
		}
	}
	//t2 = clock();
	//printf("%lf\n", (t2 - t1) / (double)(CLOCKS_PER_SEC));
	return result;
}

vector<double> SetAudioTo16Bits(vector<double> arr) {
	double maxVal = 0;
	double absArr = 0;
	for (int i = 0; i < arr.size(); i++) {
		absArr = abs(arr[i]);
		if (absArr > maxVal)
			maxVal = absArr;
	}
	if (maxVal <= 1)
		transform(arr.begin(), arr.end(), arr.begin(), bind(multiplies<double>(), placeholders::_1, pow(2, 15)));
	return arr;
}

vector<double> hz2mel(vector<double> hz) {
	vector<double> mel(hz.size(), 0);
	for (int i = 0; i < hz.size(); i++) {
		mel[i] = 1127 * log(1 + hz[i] / 700);
	}
	return mel;
}

double hz2mel(double hz) {
	return 1127 * log(1 + hz / 700);
}

vector<double> mel2hz(vector<double> mel) {
	vector<double> hz(mel.size(), 0);
	for (int i = 0; i < mel.size(); i++) {
		hz[i] = 700 * exp(mel[i] / 1127) - 700;
	}
	return hz;
}

double mel2hz(double mel) {
	return 700 * exp(mel / 1127) - 700;
}

vector<double> Preemphasize(vector<double> speech, double alpha) {
	vector<double> newSpeech(speech.size(), 0);
	newSpeech[0] = speech[0];
	for (int i = 1; i < speech.size(); i++) {
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
		y[i] = tempY + omega[n][i] * y[i+n/2];
		y[i + n / 2] = tempY - omega[n][i] * y[i+n/2];
	}
	return y;
}

vector<vector<double>> trifbank() {
	vector<vector<double>> H(k, vector<double>(numFFT, 0));
	double fl = hz2mel(lowFreq);
	double fh = hz2mel(highFreq);
	vector<double> hzVec(numFilters + 2, 0);
	for (int i = 0; i < numFilters + 2; i++) {
		hzVec[i] = fh* i* ((fh - fl) / (numFilters + 1));
	}
	vector<double> c = mel2hz(hzVec);
	vector<double> cw = hz2mel(c);

	return H;
}

int main()
{
	// Test
	/*vector<vector<double>> m1{ {2,3,4},{1,5,7} };
	vector<vector<double>> m2{ {1,6},{8,2},{9,4} };
	vector<vector<double>> m3;
	m3 = matrixMultiply(m1, m2);*/

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
	vector<vector<double>> Mag(frame[0].size(), vector<double>(numFFT, 0));
	vector<complex<double>> tempFrame;
	for (int i = 0; i < frame[0].size(); i++) {
		tempFrame.assign(frame[0].begin(), frame[0].end());
		tempFrame.resize(numFFT);
		tempFrame = FFT(tempFrame);
		for (int j = 0; j < numFFT; j++)
			Mag[i][j] = abs(tempFrame[j]);
		//SaveAsCSV(Mag[i], "test2.csv");
	}

	// Create filter bank
	vector<vector<double>> H(numFilters);
	H = trifbank();
	system("pause");
	return 0;
}