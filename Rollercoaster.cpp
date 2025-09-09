//=============================================================================================
// Zöld háromszög: A framework.h osztályait felhasználó megoldás
//=============================================================================================
#include "framework.h"

// csúcspont árnyaló
const char* vertSource = R"(
   #version 330 core
precision highp float;
layout(location = 0) in vec2 aPos;
uniform mat4 MVP;

void main()
{
    // Explicitly add a z coordinate of 0.0
    gl_Position = MVP * vec4(aPos, 0.0, 1.0);
}
)";
const char* fragSource = R"(
    #version 330
    precision highp float;

    uniform vec3 color;            // konstans sz n
    out vec4 fragmentColor;        // pixel sz n

    void main() {
        fragmentColor = vec4(color, 1); // RGB -> RGBA
    }
)";

const int winWidth = 600, winHeight = 600;
const int nTesselatedVertices = 100;

GPUProgram* gpuProgram;

class Camera2D {
	vec2 wCenter;
	vec2 wSize;
public:
	Camera2D() : wCenter(0, 0), wSize(200, 200) {}

	mat4 V() { return translate(vec3(-wCenter.x,-wCenter.y, 0)); }
	mat4 P() { return scale(vec3(2 / wSize.x, 2 / wSize.y, 1)); }

	mat4 Vinv() { return translate(vec3(wCenter.x,wCenter.y, 0)); }
	mat4 Pinv() { return scale(vec3(wSize.x / 2, wSize.y / 2, 1)); }

	void Zoom(float s) { wSize = wSize * s; }
	void Pan(vec2 t) { wCenter = wCenter + t; }
};

Camera2D camera;


class Spline {
	unsigned int vaoVectorizedCurve, vboVectorizedCurve;
	unsigned int vaoControlPoints, vboControlPoints;
protected:
	std::vector<vec2> controlPoints;
public:
	Spline() {
		
		glGenVertexArrays(1, &vaoVectorizedCurve);
		glBindVertexArray(vaoVectorizedCurve);
		glGenBuffers(1, &vboVectorizedCurve);
		glBindBuffer(GL_ARRAY_BUFFER, vboVectorizedCurve);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), NULL);
		glEnableVertexAttribArray(0);  

	
		glGenVertexArrays(1, &vaoControlPoints);
		glBindVertexArray(vaoControlPoints);
		glGenBuffers(1, &vboControlPoints);
		glBindBuffer(GL_ARRAY_BUFFER, vboControlPoints);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), NULL);
		glEnableVertexAttribArray(0); 
	}
	~Spline() {
		glDeleteBuffers(1, &vboVectorizedCurve);
		glDeleteVertexArrays(1, &vaoVectorizedCurve);
		glDeleteBuffers(1, &vboControlPoints);
		glDeleteVertexArrays(1, &vaoControlPoints);
	}
	virtual vec2 r(float t) = 0;
	virtual vec2 dr(float t) = 0;
	virtual vec2 ddr(float t) = 0;
	virtual float tStart() = 0;
	virtual float tEnd() = 0;
	virtual void AddControlPoint(float cX, float cY)
	{
		vec4 wVertex = camera.Pinv() * camera.Vinv() * vec4(cX, cY, 0, 1);
		controlPoints.push_back(vec2(wVertex.x, wVertex.y));

	}
	void Draw()
	{
		mat4 VPTransform = camera.V() * camera.P();
		gpuProgram->setUniform(VPTransform, "MVP");
	
		
		if (controlPoints.size() >= 2)
		{
			gpuProgram->Use();
			std::vector<vec2> curveVertices;
			for (int i = 0; i < nTesselatedVertices; i++)
			{
				float tNormalized = float(i) / (nTesselatedVertices - 1);
				float t = tStart() + (tEnd() - tStart()) * tNormalized;
				vec2 wVertex = r(t);
				curveVertices.push_back(wVertex);
			}
			glBindVertexArray(vaoVectorizedCurve);
			glBindBuffer(GL_ARRAY_BUFFER, vboVectorizedCurve);
			glBufferData(GL_ARRAY_BUFFER, curveVertices.size() * sizeof(vec2), &curveVertices[0], GL_DYNAMIC_DRAW);
			gpuProgram->setUniform(vec3(1, 1, 0), "color");
			glLineWidth(2.0f);
			glDrawArrays(GL_LINE_STRIP, 0, nTesselatedVertices);
		}
		if (controlPoints.size() > 0)
		{
			gpuProgram->Use();
			glBindVertexArray(vaoControlPoints);
			glBindBuffer(GL_ARRAY_BUFFER, vboControlPoints);
			glBufferData(GL_ARRAY_BUFFER, controlPoints.size() * sizeof(vec2), &controlPoints[0], GL_DYNAMIC_DRAW);
			gpuProgram->setUniform(vec3(1, 0, 0), "color");
			glPointSize(10.0f);
			glDrawArrays(GL_POINTS, 0, (int)controlPoints.size());
		}
	}
};

class CatmullRomSpline : public Spline {
	std::vector<float> knots;
	vec2 Hermite(vec2 p0, vec2 v0, float t0, vec2 p1, vec2 v1, float t1, float t)
	{
		float deltat = t1 - t0;
		t -= t0;
		float deltat2 = deltat * deltat;
		float deltat3 = deltat2 * deltat;
		vec2 a0 = p0, a1 = v0;
		vec2 a2 = (p1 - p0) * 3.0f / deltat2 - (v1 + v0 * 2.0f) / deltat;
		vec2 a3 = (p0 - p1) * 2.0f / deltat3 + (v1 + v0) / deltat2;
		return ((a3 * t + a2) * t + a1) * t + a0;
	}
	vec2 HermiteDerivative(vec2 p0, vec2 v0, float t0, vec2 p1, vec2 v1, float t1, float t) {
		float deltat = t1 - t0;
		t = (t - t0) / deltat; 
		float t2 = t * t;

		return (6 * t2 - 6 * t) * p0 +
			(3 * t2 - 4 * t + 1) * v0 * deltat +
			(-6 * t2 + 6 * t) * p1 +
			(3 * t2 - 2 * t) * v1 * deltat;
	}
	vec2 HermiteSecondDerivative(vec2 p0, vec2 v0, float t0, vec2 p1, vec2 v1, float t1, float t) {
		float deltat = t1 - t0;
		t = (t - t0) / deltat;
		float t2 = t * t;

		return (12 * t - 6) * p0 +
			(6 * t - 4) * v0 * deltat +
			(-12 * t + 6) * p1 +
			(6 * t - 2) * v1 * deltat;
	}
public:
	void AddControlPoint(float cX, float cY)
	{
		knots.push_back((float)controlPoints.size());
		Spline::AddControlPoint(cX, cY);
	}
	float tStart() { return 0; }
	float tEnd() { return knots[controlPoints.size() - 1]; }

	vec2 r(float t) override
	{
		vec2 wPoint(0, 0);
		for (int i = 0; i < controlPoints.size() - 1; i++)
		{
			if (knots[i] <= t && t <= knots[i + 1]) {
				vec2 vPrev = (i > 0) ? (controlPoints[i] - controlPoints[i - 1]) * (1.0f / (knots[i] - knots[i - 1])) : vec2(0, 0);
				vec2 vCur = (controlPoints[i + 1] - controlPoints[i]) / (knots[i + 1] - knots[i]);
				vec2 vNext = (i < controlPoints.size() - 2) ? (controlPoints[i + 2] - controlPoints[i + 1]) * (1.0f / (knots[i + 2] - knots[i + 1])) : vec2(0, 0);
				vec2 v0 = (vPrev + vCur) * 0.5f;
				vec2 v1 = (vCur + vNext) * 0.5f;
				return Hermite(controlPoints[i], v0, knots[i], controlPoints[i + 1], v1, knots[i + 1], t);
			}
		}
		return controlPoints[0];
	}
	vec2 dr(float t) override {
		if (controlPoints.size() < 2) return vec2(0);

		for (int i = 0; i < controlPoints.size() - 1; i++) {
			if (knots[i] <= t && t <= knots[i + 1]) {
				vec2 vPrev = (i > 0) ?
					(controlPoints[i] - controlPoints[i - 1]) / (knots[i] - knots[i - 1]) :
					vec2(0,0);
				vec2 vCur = (controlPoints[i + 1] - controlPoints[i]) / (knots[i + 1] - knots[i]);
				vec2 vNext = (i < controlPoints.size() - 2) ?
					(controlPoints[i + 2] - controlPoints[i + 1]) / (knots[i + 2] - knots[i + 1]) :
					vec2(0,0);

				vec2 v0 = (vPrev + vCur) * 0.5f;
				vec2 v1 = (vCur + vNext) * 0.5f;

				return HermiteDerivative(controlPoints[i], v0, knots[i],
					controlPoints[i + 1], v1, knots[i + 1], t);
			}
		}
		return vec2(0);
	}
	

	vec2 ddr(float t) override {
		if (controlPoints.size() < 2) return vec2(0, 0);

		for (int i = 0; i < controlPoints.size() - 1; i++) {
			if (knots[i] <= t && t <= knots[i + 1]) {
				vec2 vPrev = (i > 0) ?
					(controlPoints[i] - controlPoints[i - 1]) / (knots[i] - knots[i - 1]) :
					vec2(0, 0);
				vec2 vCur = (controlPoints[i + 1] - controlPoints[i]) / (knots[i + 1] - knots[i]);
				vec2 vNext = (i < controlPoints.size() - 2) ?
					(controlPoints[i + 2] - controlPoints[i + 1]) / (knots[i + 2] - knots[i + 1]) :
					vec2(0, 0);

				vec2 v0 = (vPrev + vCur) * 0.5f;
				vec2 v1 = (vCur + vNext) * 0.5f;

				return HermiteSecondDerivative(controlPoints[i], v0, knots[i],
					controlPoints[i + 1], v1, knots[i + 1], t);
			}
		}
		return vec2(0, 0);
	}
};
Spline* spline;
enum GondolaState { WAITING, STARTED, FALLEN };
class Gondola {
private:
	Geometry<vec2> circleGeom;
	Geometry<vec2> plusGeom;
public:
	Spline* spline;
	float tau;
	float v;
	GondolaState state;
	mat4 modelMatrix;
	float rollAngle;  
	float curvature;
	const float g = 40.0f;      
	const float lambda = 0.5f;  
	const float radius = 10.0f;
	Gondola(Spline* _spline) : spline(_spline), state(WAITING), modelMatrix(1.0f)
	{
		circleGeom.Vtx().push_back(vec2(0, 0));
		for (int i = 0; i < 101; i++)
		{
			float angle = 2.0f * M_PI * i / 100.0f;
			float x = radius * cos(angle);
			float y = radius * sin(angle);
			circleGeom.Vtx().push_back(vec2(x, y));
		}
		circleGeom.updateGPU();	


		plusGeom.Vtx().push_back(vec2(-radius, 0));
		plusGeom.Vtx().push_back(vec2(radius, 0));
		plusGeom.Vtx().push_back(vec2(0, -radius));
		plusGeom.Vtx().push_back(vec2(0, radius));
		plusGeom.updateGPU();
	
	}
	GondolaState getState() {
		return this->state;
	}
	void Start() {
		tau = 0.01f; 
		v = 0.0f;   
		state = STARTED; 
		rollAngle = 0.0f; 
	}
	void Animate(float dt) {
		
		if (state != STARTED) return;

		float heightDiff = std::max(0.0f, spline->r(0).y - spline->r(tau).y);
		v = sqrt((2.0f * g * heightDiff) / (1.0f + lambda)) * 2.5;
		
		vec2 tauPos = spline->r(tau);
		vec2 tangent = normalize(spline->dr(tau));

		float sqrtOfDerivative = sqrt(dot(spline->dr(tau), spline->dr(tau)));
		float deltaTau = (v * dt) / sqrtOfDerivative;
		tau += deltaTau;

		vec2 pos = spline->r(tau);

		vec2 normal = vec2(-tangent.y, tangent.x);
		vec2 gondolaCenter = pos + normal * radius;

		float ds = -v * dt;
		rollAngle += ds / radius; 

		modelMatrix = translate(mat4(1.0f), vec3(gondolaCenter.x, gondolaCenter.y, 0.0f)) *
			rotate(mat4(1.0f), rollAngle, vec3(0.0f, 0.0f, 1.0f));

		vec2 firstDeriv = spline->dr(tau);
		float firstderivLength = sqrt(pow(firstDeriv.x, 2) + pow(firstDeriv.y, 2));
		float curvature = dot(spline->ddr(tau), tangent) / pow(firstderivLength, 2);
		
		vec2 gravitationalForce = vec2(0.0f, -g);
		vec2 centripetalForce = curvature * v * v * normal; 
		vec2 K_vector = gravitationalForce + centripetalForce; 

		float K_magnitude = vectorLength(K_vector);
		float normalMagnitude = vectorLength(normal); 
	
		if (K_magnitude < normalMagnitude) {
			state = FALLEN;
		}
		if (v == 0.0f) {
			tau = 0.01f;
			return;
		}
	}

	float vectorLength(vec2 vec)
	{
		return sqrt(pow(vec.x, 2) + pow(vec.y, 2));
	}
	void Draw()
	{
		gpuProgram->Use();
		mat4 viewProj = camera.V() * camera.P();
		mat4 mvp = viewProj * modelMatrix;
		gpuProgram->setUniform(mvp, "MVP");

		circleGeom.Draw(gpuProgram, GL_TRIANGLE_FAN, vec3(0.0f, 0.0f, 1.0f));
		circleGeom.Draw(gpuProgram, GL_LINE_STRIP, vec3(1.0f, 1.0f, 1.0f));
		 glLineWidth(4.0f);
		 gpuProgram->setUniform(mvp, "MVP");
        plusGeom.Draw(gpuProgram, GL_LINES, vec3(1.0f, 1.0f, 1.0f));
	}



};


Gondola* gondola;

class MyWindow : public glApp {
public:
	MyWindow() : glApp("Points And Lines") {}

	void onInitialization() {
		glViewport(0, 0, winWidth, winHeight);
		glLineWidth(2.0f);

		gpuProgram = new GPUProgram(vertSource, fragSource);
		spline = new CatmullRomSpline();
		gondola = new Gondola(spline);
	}
	void onDisplay() {
		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT);
		
		if (gondola->getState() == GondolaState::STARTED)
		{
			gondola->Draw();
		}
		spline->Draw();
	
		
	}
	void onMousePressed(MouseButton button, int pX, int pY)
	{
		float cX = 2.0f * pX / winWidth - 1;
		float cY = 1.0f - 2.0f * pY / winHeight;
		printf("Mouse pressed at %f %f\n", cX, cY);
		spline->AddControlPoint(cX, cY);
		refreshScreen();
	}
	void onKeyboard(int key) {
		printf("Key pressed: %d\n", key);
		if (key == 32) { 
			gondola->Start();
			
		}
		refreshScreen();
	}

	void onMouseMotion() {

	}
	void onIdle()
	{

	}
	void onTimeElapsed(float tstart, float tend) {
		if (gondola->getState() == GondolaState::STARTED)
		{
			const float dt = 0.01; 
			for (float t = tstart; t < tend; t += dt) {
				float Dt = fmin(dt, tend - t);
				gondola->Animate(Dt);
			}
			refreshScreen();
		}
		
	}
};

MyWindow app;