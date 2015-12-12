#ifndef Course_h
#define Course_h

#include <string>

class Course {
 public:
  // Default constructor
  Course();

  // Constructor
  Course(std::string nm, int gr);

  // method to get the name of the course
  std::string getCourseName();

  // method to get the grade of the course
  int getCourseGrade();

 private:
  std::string course_name;      // name of this course
  int grade;                    // grade of this course
};
#endif
