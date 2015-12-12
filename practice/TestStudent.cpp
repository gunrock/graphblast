#include "TestStudent.h"
#include "stdlib.h"

using namespace CppUnit;

// method to test the constructor
void StudentTestCase::testConstructor() {  // Note 4
  // create a student object
  Student stu("Tan Meng Chee", "94-1111B-13");

  // check that the object is constructed correctly - Note 5
  std::string student_name = stu.getStuName();
  CPPUNIT_ASSERT(student_name == "Tan Meng Chee");
  std::string student_number = stu.getStuNumber();
  CPPUNIT_ASSERT(student_number == "94-1111B-13");
}

// method to test the assigning and retrieval of grades
void StudentTestCase::testAssignAndRetrieveGrades() {
  // create a student
  Student stu("Jimmy", "946302B");

  // assign a few grades to this student
  stu.assignGrade("cs2102", 60);
  stu.assignGrade("cs2103", 70);
  stu.assignGrade("cs3214s", 80);

  // verify that the assignment is correct - Note 6
  CPPUNIT_ASSERT_EQUAL(60, stu.getGrade("cs2102"));
  CPPUNIT_ASSERT_EQUAL(70, stu.getGrade("cs2103"));
        
  // attempt to retrieve a course that does not exist
  CPPUNIT_ASSERT_EQUAL(-1, stu.getGrade("cs21002"));
}

// method to create a suite of tests - Note 7
Test *StudentTestCase::suite () {
  TestSuite *testSuite = new TestSuite ("StudentTestCase");
  
  // add the tests
  testSuite->addTest (new TestCaller<StudentTestCase>  
      ("testConstructor", &StudentTestCase::testConstructor));
  testSuite->addTest (new TestCaller<StudentTestCase> 
      ("testAssignAndRetrieveGrades", 
       &StudentTestCase::testAssignAndRetrieveGrades));
  return testSuite;
}

// the main method - Note 8
int main (int argc, char* argv[]) {
  if (argc != 2) {
    std::cout << "usage: tester name_of_class_being_test" << std::endl;
    exit(1);
  }

  // informs test-listener about testresults
  TestResult result;

  TestRunner runner;
  //runner.addTest(argv[1], StudentTestCase::suite());
  //runner.run(argc, argv)

  runner.addTest(StudentTestCase::suite());
  runner.run(result);
  
  return 0;
}
