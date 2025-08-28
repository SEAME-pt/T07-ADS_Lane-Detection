NAME	= zombie00

CXX		= c++

CXXFLAGS	= -Wall -Werror -Wextra -std=c++98

SRC		= main.cpp	Zombie.cpp	randomChump.cpp		newZombie.cpp

OBJ		= $(SRC:.cpp=.o)

RM		= rm -rf

all: $(NAME)

$(NAME): $(OBJ) 
	$(CXX) $(CXXFLAGS) $(SRC) -o $(NAME)

clean:
	$(RM) $(OBJ)

fclean: clean
	$(RM) $(NAME)
	
re:	 fclean all

.PHONY:	all clean fclean re bonus
