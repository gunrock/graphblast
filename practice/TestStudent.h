#ifndef TestStudent_h
#define TestStudent_h

#include <iostream>
#include <string>

// Note 1
#include "TestCase.h"
#include "TestSuite.h"
#include "TestCaller.h"
#include "TestRunner.h"
#include "TestResult.h"

#include "Student.h"

using namespace CppUnit;

class StudentTestCase : public TestCase { // Note 2 
public:
  // constructor - Note 3
  StudentTestCase(std::string name) : TestCase(name) {}

  // method to test the constructor
  void testConstructor();

  // method to test the assigning and retrieval of grades
  void testAssignAndRetrieveGrades();

  // method to create a suite of tests
  static Test *suite ();
};
#endif
